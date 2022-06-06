
import           Data.List (foldl')
import qualified Network.Socket as Net
import qualified System.Console.GetOpt as Opt
import           System.Environment (getProgName, getArgs)
import           System.Exit (exitFailure)
import           System.IO (hPutStrLn, stderr)
import           System.IO.Unsafe (unsafePerformIO)

import qualified Network.KVS.Server as KVS

data Options = Options
  { optionHost :: Net.HostAddress
  , optionPort :: Net.PortNumber
  }

defaultOptions :: Options
defaultOptions = Options
  { optionHost = Net.iNADDR_ANY
  , optionPort = Net.aNY_PORT
  }

options :: [Opt.OptDescr (Options -> Options)]
options =
  [ Opt.Option "H" ["host"] (Opt.ReqArg (\h o -> o{ optionHost = unsafePerformIO (Net.inet_addr h) }) "HOST")
    "Host interface"
  , Opt.Option "p" ["port"] (Opt.ReqArg (\p o -> o{ optionPort = read p }) "PORT")
    "Port [random]"
  ]

main :: IO ()
main = do
  prog <- getProgName
  args <- getArgs
  opt <- case Opt.getOpt Opt.Permute options args of
    (ol, [], []) -> 
      return $ foldl' (flip ($)) defaultOptions ol
    (_, _, err) -> do
      mapM_ (hPutStrLn stderr) err
      hPutStrLn stderr $ Opt.usageInfo ("Usage: " ++ prog ++ " [OPTIONS]") options
      exitFailure

  KVS.serve (Net.SockAddrInet (optionPort opt) (optionHost opt))
