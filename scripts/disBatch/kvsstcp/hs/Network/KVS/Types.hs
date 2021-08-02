{-# LANGUAGE OverloadedStrings #-}
module Network.KVS.Types
  ( Key
  , Value
  , Encoding
  , fromEncoding
  , toEncoding
  , defaultEncoding
  , EncodedValue
  , EventType(..)
  , eventName
  , eventChar
  , parseEvents
  ) where

import qualified Data.ByteString as BS
import           Data.ByteString.Internal (c2w)
import           Data.String (IsString(..))
import           Data.Word (Word8)

type Key = BS.ByteString
type Value = BS.ByteString

-- |Encodings have exactly 4 characters.
data Encoding = Encoding !Word8 !Word8 !Word8 !Word8

fromEncoding :: Encoding -> BS.ByteString
fromEncoding (Encoding a b c d) = BS.pack [a,b,c,d]

instance Show Encoding where
  showsPrec p = showsPrec p . fromEncoding

toEncoding :: BS.ByteString -> Encoding
toEncoding s = case BS.unpack s of
  [a,b,c,d] -> Encoding a b c d
  _ -> error "Network.KVS.Encoding: invalid"

instance IsString Encoding where
  fromString [a,b,c,d] = Encoding (c2w a) (c2w b) (c2w c) (c2w d)
  fromString s = error $ "Network.KVS.Encoding invalid: " ++ show s

-- |Default encoding (@"ASTR"@)
defaultEncoding :: Encoding
defaultEncoding = "ASTR"

type EncodedValue = (Encoding, Value)

data EventType
  = EventGet
  | EventPut
  | EventView
  | EventWait
  deriving (Eq, Ord, Bounded, Enum)

eventName :: EventType -> BS.ByteString
eventName EventGet = "get"
eventName EventPut = "put"
eventName EventView = "view"
eventName EventWait = "wait"

eventChar :: EventType -> Word8
eventChar = BS.head . eventName

charEvent :: Char -> Maybe EventType
charEvent 'g' = Just EventGet
charEvent 'p' = Just EventPut
charEvent 'v' = Just EventView
charEvent 'w' = Just EventWait
charEvent _ = Nothing

parseEvents :: String -> Maybe [EventType]
parseEvents = mapM charEvent
