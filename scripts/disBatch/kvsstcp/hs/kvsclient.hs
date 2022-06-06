
import           Control.Arrow (first)
import           Control.Concurrent (threadDelay)
import           Control.Monad (guard)
import qualified Data.ByteString.Char8 as BSC
import           Data.Fixed (Fixed(MkFixed), Micro)
import           Data.List (foldl', elemIndices)
import           Data.String (fromString)
import qualified Network.Socket as Net
import qualified System.Console.GetOpt as Opt
import           System.Environment (getProgName, getArgs)
import           System.Exit (exitFailure)
import           System.IO (hPutStrLn, stderr)

import           Network.KVS.Types
import qualified Network.KVS.Client as KVS

-- |Split a string on the last colon
splitColon :: String -> Maybe (String, String)
splitColon "" = Nothing
splitColon (':':s) = Just $ maybe ("", s) (first (':':)) $ splitColon s
splitColon (c:s) = first (c:) <$> splitColon s

data Options = Options
  { optionServer :: Maybe (Net.HostName, Net.PortNumber)
  , optionEncoding :: Encoding
  , optionArgument :: Maybe (String -> Options -> Options)
  , optionOps :: KVS.KVSClient -> IO ()
  }

defaultOptions :: Options
defaultOptions = Options
  { optionServer = Nothing
  , optionEncoding = defaultEncoding
  , optionArgument = Nothing
  , optionOps = \_ -> return ()
  }

noArg :: Options -> Options
noArg Options{ optionArgument = Just _ } = error "Missing argument"
noArg opts = opts

options :: [Opt.OptDescr (Options -> Options)]
options =
  [ Opt.Option "E" ["encoding"] (Opt.ReqArg (\e o -> o{ optionEncoding = fromString e }) "CODE")
    "Value encoding to set (4-character string) [ASTR]"
  , Opt.Option "p" ["put"] (Opt.ReqArg (\k -> addArg $ \v -> addOp $ \o s ->
      KVS.put s (fromString k) (optionEncoding o, BSC.pack v)) "KEY VALUE")
    "Put a value"
  , Opt.Option "g" ["get"] (Opt.ReqArg (\k -> addOp $ \_ s ->
      print =<< KVS.get s (fromString k)) "KEY")
    "Retrieve and remove a value"
  , Opt.Option "v" ["view"] (Opt.ReqArg (\k -> addOp $ \_ s ->
      print =<< KVS.view s (fromString k)) "KEY")
      "Retrieve a value"
  , Opt.Option "m" ["monkey"] (Opt.ReqArg (\m -> addArg $ \ke ->
    maybe (error "Invalid monitor specification")
      (\(k, e) -> addOp $ \_ s ->
        KVS.monkey s (fromString m) (fromString k <$ guard (not $ null k)) e)
      $ do
        (k, e) <- splitColon ke
        (,) k <$> parseEvents e) "MKEY [KEY]:EVENTS")
      "Create or update a monitor for the key and events"
  , Opt.Option "S" ["shutdown"] (Opt.NoArg (addOp $ \_ s ->
      KVS.shutdown s))
      "Tell the server to shutdown"
  , Opt.Option "s" ["sleep"] (Opt.ReqArg (\t -> addOp $ \_ _ ->
      threadDelay $ micros $ read t) "SECS")
      "Pause for a time"
  ] where
  addArg f opts = (noArg opts){ optionArgument = Just f }
  addOp op opts = opts{ optionOps = \s -> optionOps opts s >> op opts s }
  micros :: Micro -> Int
  micros (MkFixed i) = fromInteger i

argument :: String -> Options -> Options
argument arg opt@Options{ optionArgument = Just f } =
  f arg opt{ optionArgument = Nothing }
argument arg opt
  | Nothing <- optionServer opt
  , colons@(_:_) <- elemIndices ':' arg
  , (h, ~(':':p)) <- splitAt (last colons) arg =
    opt{ optionServer = Just (h, read p) }
  | otherwise = error $ "Unhandled argument: " ++ show arg

main :: IO ()
main = do
  prog <- getProgName
  args <- getArgs
  opt <- case Opt.getOpt (Opt.ReturnInOrder argument) options args of
    (ol, [], []) -> 
      return $ noArg $ foldl' (flip ($)) defaultOptions ol
    (_, _, err) -> do
      mapM_ (hPutStrLn stderr) err
      hPutStrLn stderr $ Opt.usageInfo ("Usage: " ++ prog ++ " [HOST:PORT] OPS...") options
      exitFailure

  s <- KVS.connect (optionServer opt)
  optionOps opt s
  KVS.close s
