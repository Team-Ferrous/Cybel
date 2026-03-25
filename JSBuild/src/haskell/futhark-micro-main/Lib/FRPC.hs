{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module FRPC
  ( diffNoAlphaImages
  , diffImages
  , blurImage
  , resizeImage
  , FRPC.Manager
  , FRPC.Server
  -- add more as needed
  ) where

import Futhark           (FutIO, toFuthark, fromFuthark)
import qualified Massiv.Array as MP
import Massiv.Array as MP (Image, RGBA(..))

-- ============================================================
-- Placeholder stubs for Futhark GPU functions
-- ============================================================

-- | Diff two images ignoring alpha channel
diffNoAlphaImages :: MP.Image MP.RGBA Float -> MP.Image MP.RGBA Float -> FutIO (MP.Image MP.RGBA Float)
diffNoAlphaImages imgA imgB = do
    futA <- toFuthark $ unImage imgA
    futB <- toFuthark $ unImage imgB
    futResult <- futhark_diffNoAlpha futA futB
    fromFuthark futResult

-- | Full diff including alpha channel
diffImages :: MP.Image MP.RGBA Float -> MP.Image MP.RGBA Float -> FutIO (MP.Image MP.RGBA Float)
diffImages imgA imgB = do
    futA <- toFuthark $ unImage imgA
    futB <- toFuthark $ unImage imgB
    futResult <- futhark_diff futA futB
    fromFuthark futResult

-- | Simple blur kernel placeholder
blurImage :: MP.Image MP.RGBA Float -> FutIO (MP.Image MP.RGBA Float)
blurImage img = do
    futImg <- toFuthark $ unImage img
    futResult <- futhark_blur futImg
    fromFuthark futResult

-- | Resize kernel placeholder
resizeImage :: MP.Image MP.RGBA Float -> Int -> Int -> FutIO (MP.Image MP.RGBA Float)
resizeImage img newH newW = do
    futImg <- toFuthark $ unImage img
    futResult <- futhark_resize futImg newH newW
    fromFuthark futResult

-- ============================================================
-- Futhark FFI stubs (replace with real Futhark bindings)
-- ============================================================

-- Each of these would eventually be replaced by proper FFI imports
-- or by using your compiled Futhark library

futhark_diffNoAlpha :: a -> a -> FutIO a
futhark_diffNoAlpha = undefined

futhark_diff :: a -> a -> FutIO a
futhark_diff = undefined

futhark_blur :: a -> FutIO a
futhark_blur = undefined

futhark_resize :: a -> Int -> Int -> FutIO a
futhark_resize = undefined

-- ============================================================
-- Helpers
-- ============================================================

-- Convert MP.Image to underlying array for Futhark
unImage :: MP.Image MP.RGBA Float -> MP.Array MP.U MP.RGBA Float
unImage = id  -- in real code, convert to contiguous representation if needed
