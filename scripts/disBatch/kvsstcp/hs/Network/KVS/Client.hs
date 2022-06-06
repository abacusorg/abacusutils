{-# LANGUAGE OverloadedStrings #-}
module Network.KVS.Client
  ( KVSClient
  , connect
  , close
  , shutdown
  , put
  , get
  , view
  , monkey
  ) where

import           Control.Exception (finally)
import           Control.Monad (when)
import           Control.Arrow (second)
import qualified Data.ByteString as BS
import           Data.ByteString.Internal (c2w)
import           Data.Foldable (fold)
import           Data.Monoid ((<>))
import qualified Network.Socket as Net
import qualified Network.Socket.ByteString as NetBS
import           System.Environment (getEnv)
import           System.IO.Error (catchIOError, tryIOError)

import           Network.KVS.Types
import           Network.KVS.Internal

-- |A connection to a KVS server.
-- Connections are NOT thread-safe: only one operation may be active at a time.
newtype KVSClient = KVSClient
  { _kvsSocket :: Net.Socket
  }

getEnvHostPort :: IO (String, String)
getEnvHostPort = (,) <$> getEnv "KVSSTCP_HOST" <*> getEnv "KVSSTCP_PORT"

connectAddr :: Net.AddrInfo -> IO Net.Socket
connectAddr a = do
  s <- Net.socket (Net.addrFamily a) (Net.addrSocketType a) (Net.addrProtocol a)
  Net.setSocketOption s Net.NoDelay 1
  Net.connect s (Net.addrAddress a)
  return s

tryConnectEach :: [Net.AddrInfo] -> IO Net.Socket
tryConnectEach [] = fail "No addresses available"
tryConnectEach [a] = connectAddr a
tryConnectEach (a:l) = connectAddr a `catchIOError` \_ -> tryConnectEach l

-- |Establish connection to a KVS server at the given address, or @($KVSSTCP_HOST, $KVSSTCP_PORT)@ from the environment if not supplied.
connect :: Maybe (Net.HostName, Net.PortNumber) -> IO KVSClient
connect hp = do
  (host, port) <- maybe getEnvHostPort (return . second show) hp
  ai <- Net.getAddrInfo (Just Net.defaultHints
    { Net.addrFlags = [Net.AI_NUMERICSERV]
    , Net.addrSocketType = Net.Stream
    }) (Just host) (Just port)
  KVSClient <$> tryConnectEach ai

-- |Close the connection to the KVS storage server. Does a socket shutdown as well.
close :: KVSClient -> IO ()
close (KVSClient s) = do
  c <- Net.isConnected s
  when c $ do
    _ <- tryIOError $ do
      NetBS.sendAll s "clos"
      Net.shutdown s Net.ShutdownBoth
    Net.close s

-- |Tell the KVS server to shutdown (and close the connection).
shutdown :: KVSClient -> IO ()
shutdown kvs@(KVSClient s) =
  NetBS.sendAll s "down" `finally` close kvs

-- |Add a value to the key.
put :: KVSClient -> Key -> EncodedValue -> IO ()
put (KVSClient s) k ev = do
  NetBS.sendAll s "put_"
  sendLenBS s k
  sendEncodedValue s ev

getView :: BS.ByteString -> KVSClient -> Key -> IO EncodedValue
getView op (KVSClient s) k = do
  NetBS.sendAll s op
  sendLenBS s k
  recvEncodedValue s

-- |Retrieve and remove a value from the store.  If there is no value associated with this key, block until one is added by another client (with 'put').
get :: KVSClient -> Key -> IO EncodedValue
get = getView "get_"

-- |Retrieve, but do not remove, a value from the store.  See 'get'.
view :: KVSClient -> Key -> IO EncodedValue
view = getView "view"

-- |Make key a monitor key, specifying what events to monitor and for which key.
-- Whenever a listed event occurs for the second key, a put will be done to the first key with the value "<event> <key>".  If key is Nothing, the events listed will be monitored for all keys.  Monitoring of any event /not/ listed is turned off for the specified key.
monkey :: KVSClient -> Key -> Maybe Key -> [EventType] -> IO ()
monkey (KVSClient s) m k e = do
  NetBS.sendAll s "mkey"
  sendLenBS s m
  sendLenBS s $ fold k <> BS.pack (c2w ':':map eventChar e)
