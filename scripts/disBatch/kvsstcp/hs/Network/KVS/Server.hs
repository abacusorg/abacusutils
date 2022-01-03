{-# LANGUAGE CPP #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE ViewPatterns #-}

module Network.KVS.Server
  ( serve
  ) where

import           Control.Arrow (first)
import           Control.Concurrent (ThreadId, myThreadId, forkIOWithUnmask, throwTo, killThread, threadWaitRead)
import           Control.Concurrent.MVar (MVar, newMVar, modifyMVar, modifyMVar_, readMVar)
import           Control.Exception (Exception(..), SomeException, AsyncException, asyncExceptionToException, asyncExceptionFromException, catch, handle, try, mask, mask_)
import           Control.Monad (when, forever, guard)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BSC
import           Data.List (foldl')
import           Data.Monoid ((<>))
#ifdef VERSION_unordered_containers
import           Data.Hashable (Hashable)
import qualified Data.HashMap.Strict as Map
import qualified Data.HashSet as Set
#else
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
#endif
import qualified Network.Socket as Net
#ifdef VERSION_unix
import           System.Posix.Resource (getResourceLimit, setResourceLimit, Resource(ResourceOpenFiles), softLimit, hardLimit, ResourceLimit(..))
#endif
import           System.Posix.Types (Fd(Fd))
import           System.IO (hPutStrLn, stderr)

import           Network.KVS.Types
import           Network.KVS.Internal
import           Network.KVS.MQueue

#ifdef VERSION_unix
instance Ord ResourceLimit where
  ResourceLimit x <= ResourceLimit y = x <= y
  ResourceLimitUnknown <= _ = True
  _ <= ResourceLimitUnknown = False
  _ <= ResourceLimitInfinity = True
  ResourceLimitInfinity <= _ = False
#endif

#ifdef VERSION_unordered_containers
type Map = Map.HashMap
type Set = Set.HashSet
#else
type Map = Map.Map
type Set = Set.Set
#endif

logger :: String -> IO ()
logger = hPutStrLn stderr

data StopWaiting = StopWaiting
  deriving (Show)
instance Exception StopWaiting where
  toException = asyncExceptionToException
  fromException = asyncExceptionFromException

data EventMap a = EventMap
  { eventGet, eventPut, eventView, eventWait :: a }

instance Functor EventMap where
  fmap f (EventMap g p v w) =
    EventMap (f g) (f p) (f v) (f w)

instance Applicative EventMap where
  pure a = EventMap a a a a
  EventMap fg fp fv fw <*> EventMap g p v w =
    EventMap (fg g) (fp p) (fv v) (fw w)

instance Monoid a => Monoid (EventMap a) where
  mempty = pure mempty
  mappend (EventMap g1 p1 v1 w1) (EventMap g2 p2 v2 w2) =
    EventMap (mappend g1 g2) (mappend p1 p2) (mappend v1 v2) (mappend w1 w2)

lookupEvent :: EventType -> EventMap a -> a
lookupEvent EventGet = eventGet
lookupEvent EventPut = eventPut
lookupEvent EventView = eventView
lookupEvent EventWait = eventWait

insertEvent :: EventType -> a -> EventMap a -> EventMap a
insertEvent EventGet v e = e{ eventGet = v }
insertEvent EventPut v e = e{ eventPut = v }
insertEvent EventView v e = e{ eventView = v }
insertEvent EventWait v e = e{ eventWait = v }

splitEvents :: BS.ByteString -> Maybe (Maybe BS.ByteString, [EventType])
splitEvents s = do
  c <- BSC.elemIndexEnd ':' s
  let (k, e) = BS.splitAt c s
  (k <$ guard (not $ BS.null k), ) <$> parseEvents (BSC.unpack e)

type Monitors = EventMap (Set Key)

data Entry = Entry
  { _entryMQueue :: MQueue EncodedValue
  , _entryMonitor :: Monitors
  }

data KVS = KVS
  { _kvsMap :: Map Key Entry
  , _kvsMonitor :: Monitors -- ^global monitors
  }

lookupEntry :: MVar KVS -> BS.ByteString -> IO Entry
lookupEntry mkvs k = modifyMVar mkvs $ \kvs@(KVS m gmon) -> maybe
  (do
    q <- newMQueue
    let e = Entry q mempty
    return (KVS (Map.insert k e m) gmon, Entry q gmon))
  (\(Entry q emon) ->
    return (kvs, Entry q (emon <> gmon)))
  $ Map.lookup k m

setMonkey :: BS.ByteString -> (Maybe BS.ByteString, [EventType]) -> KVS -> IO KVS
setMonkey mkey (key, events) (KVS m gmon) = maybe
  (return $ KVS m (setev gmon))
  (\k -> do
    q <- newMQueue
    let e = Entry q (setev mempty)
    return $ KVS (Map.insertWith (\_ (Entry eq emon) -> Entry eq (setev emon)) k e m) gmon)
  key
  where
  setev = (foldl' (\em e -> insertEvent e (Set.insert mkey) em) (pure $ Set.delete mkey) events <*>)

doMonkeys :: MVar KVS -> (EventType, BS.ByteString) -> Monitors -> IO ()
doMonkeys mkvs evk@(event, _) = do
  mapM_ (\k -> do
      Entry q _ <- lookupEntry mkvs k
      putMQueue evkv q)
    . lookupEvent event
  where
  evkv = (defaultEncoding, BSC.pack $ show $ first eventName evk)

client :: MVar KVS -> Net.Socket -> Net.SockAddr -> ThreadId -> IO Bool
client mkvs s _a tid = loop where
  loop = recvAll s 4 >>= cmd
  cmd "put_" = do
    key <- recvLenBS s
    Entry q m <- lookupEntry mkvs key
    val <- recvEncodedValue s
    putMQueue val q
    doMonkeys mkvs (EventPut, key) m
    loop
  cmd "get_" = do
    key <- recvLenBS s
    Entry q m <- lookupEntry mkvs key
    val <- wait $ getMQueue q
    mapM_ (sendEncodedValue s) val
    loop
  cmd "view" = do
    key <- recvLenBS s
    Entry q m <- lookupEntry mkvs key
    val <- wait $ viewMQueue q
    mapM_ (sendEncodedValue s) val
    loop
  cmd "mkey" = do
    mkey <- recvLenBS s
    val <- recvLenBS s
    mapM_ (modifyMVar_ mkvs . setMonkey mkey) $ splitEvents val
    loop
  cmd "dump" = do
    fail "TODO"
    loop
  cmd "clos" = do
    Net.shutdown s Net.ShutdownBoth
    return False
  cmd "down" =
    return True
  cmd c = fail $ "unknown op: " ++ show c
  getkeyq = lookupEntry mkvs =<< recvLenBS s
  wait f = mask $ \unmask -> do
    wid <- forkIOWithUnmask ($ do
      threadWaitRead $ Fd $ Net.fdSocket s
      throwTo tid StopWaiting)
    -- we don't explicitly unmask f: instead only allow interruptions while blocked, otherwise defer the StopWaiting
    r <- (Just <$> f) `catch` \StopWaiting -> return Nothing
    -- if we got stopped after f's wait completed, just ignore it
    handle (\StopWaiting -> return ()) $ unmask $ killThread wid
    return r
    
serve :: Net.SockAddr -> IO ()
serve sa = do
#ifdef VERSION_unix
  nof <- getResourceLimit ResourceOpenFiles
  let tnof = min (hardLimit nof) (ResourceLimit 4096)
  when (softLimit nof < tnof) $
    setResourceLimit ResourceOpenFiles nof{ softLimit = tnof }
#endif

  clients <- newMVar Set.empty
  kvs <- newMVar $ KVS Map.empty mempty
  
  s <- Net.socket (case sa of
    Net.SockAddrInet{} -> Net.AF_INET
    Net.SockAddrInet6{} -> Net.AF_INET6
    Net.SockAddrUnix{} -> Net.AF_UNIX
    Net.SockAddrCan{} -> Net.AF_CAN) Net.Stream Net.defaultProtocol
  Net.setSocketOption s Net.ReuseAddr 1
  Net.bind s sa
  a <- Net.getSocketName s
  logger $ "Server running at " ++ show a
  Net.listen s 4000

  pid <- myThreadId
  handle (\e -> logger $ "Shutting down: " ++ show (e :: AsyncException)) $ mask_ $ forever $ do
    (c, ca) <- Net.accept s
    logger $ "Acceped connect from " ++ show ca
    Net.setSocketOption c Net.NoDelay 1
    tid <- forkIOWithUnmask $ \unmask -> do
      tid <- myThreadId
      r <- try $ unmask $ client kvs c ca tid
      modifyMVar_ clients $ return . Set.delete tid
      unmask $ do
        logger $ "Closing connection from " ++ show ca ++ either (\e -> ": " ++ show (e :: SomeException)) (const "") r
        Net.close c
        when (either (const False) id r) $ killThread pid
    modifyMVar_ clients $ return . Set.insert tid

  mapM_ killThread =<< readMVar clients
