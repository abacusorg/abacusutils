module Network.KVS.Internal
  ( -- * IO
    recvAll

    -- ** Data blocks
  , sendLenBS
  , recvLenBS

    -- ** Encoded Values
  , sendEncodedValue
  , recvEncodedValue
  ) where

import           Control.Monad (guard, unless)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Builder as BSB
import qualified Data.ByteString.Char8 as BSC
import qualified Data.ByteString.Lazy as BSL
import           Data.Monoid ((<>))
import           Data.Word (Word8)
import qualified Network.Socket as Net
import qualified Network.Socket.ByteString as NetBS
import           Foreign.Marshal.Alloc (allocaBytes)
import           Foreign.Ptr (Ptr, plusPtr, castPtr)
import           System.IO.Error (ioError, mkIOError, eofErrorType)

import           Network.KVS.Types

eof :: IO a
eof = ioError $ mkIOError eofErrorType "Network.KVS.recv" Nothing Nothing

recvBufAll :: Net.Socket -> Ptr Word8 -> Int -> IO Bool
recvBufAll _ _ 0 = return True
recvBufAll s p n = do
  l <- Net.recvBuf s p n
  if l == 0
    then return False
    else recvBufAll s (p `plusPtr` l) (n - l)

recvAll :: Net.Socket -> Int -> IO BS.ByteString
recvAll s n = allocaBytes n $ \p -> do
  r <- recvBufAll s p n
  unless r eof
  BS.packCStringLen (castPtr p, n)

-- |Number of characters (decimal digits) in the length header, which may be space padded
lenChars :: Int
lenChars = 10

sendLen :: Net.Socket -> Int -> IO ()
sendLen s l
  | c < 0 = fail "Network.KVS: data too long"
  | otherwise = NetBS.sendAll s $ BSC.replicate c ' ' <> d
  where
  d = BSL.toStrict $ BSB.toLazyByteString $ BSB.intDec l
  c = lenChars - BS.length d

-- |Send a length header followed by a block of data.
sendLenBS :: Net.Socket -> BS.ByteString -> IO ()
sendLenBS s b = sendLen s (BS.length b) >> NetBS.sendAll s b

recvLen :: Net.Socket -> IO Int
recvLen s = parseLen =<< recvAll s lenChars where
  parseLen l = maybe (fail $ "Network.KVS: could not parse length: " ++ show l) return
    $ (checkParse =<<) $ BSC.readInt $ BSC.dropWhile (' ' ==) l
  checkParse (n, r) = n <$ guard (BS.null r)

-- |Receive a length header followed by a block of data.
-- Throws an EOF IO error on connection closed.
recvLenBS :: Net.Socket -> IO BS.ByteString
recvLenBS s = recvLen s >>= recvAll s

sendEncoding :: Net.Socket -> Encoding -> IO ()
sendEncoding s = NetBS.sendAll s . fromEncoding

recvEncoding :: Net.Socket -> IO Encoding
recvEncoding s = toEncoding <$> recvAll s 4

sendEncodedValue :: Net.Socket -> EncodedValue -> IO ()
sendEncodedValue s (e, v) = do
  sendEncoding s e
  sendLenBS s v

recvEncodedValue :: Net.Socket -> IO EncodedValue
recvEncodedValue s = (,) <$> recvEncoding s <*> recvLenBS s
