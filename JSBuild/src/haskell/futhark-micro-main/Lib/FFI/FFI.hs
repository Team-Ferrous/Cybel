{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}

module FRPC.FFI
  ( diffNoAlphaImages
  , diffImages
  , blurImage
  , resizeImage
  ) where

import Futhark (Context, FutIO, getContext, liftIO)
import qualified Massiv.Array as MP
import Massiv.Array (Image, RGBA(..))
import Foreign
import Foreign.C.Types
import Foreign.Marshal.Array

--------------------------------------------------------------------------------
-- | Generic helper to wrap a Futhark kernel that returns a new image
-- Input images are flattened and passed as Ptr CFloat
-- Output image is allocated, the kernel fills it
--------------------------------------------------------------------------------
withTwoImages :: (Ptr () -> Ptr CFloat -> Ptr CFloat -> CInt -> CInt -> IO CInt)
              -> MP.Image MP.RGBA Float
              -> MP.Image MP.RGBA Float
              -> FutIO (MP.Image MP.RGBA Float)
withTwoImages ffiFunc imgA imgB = do
    let h = fromIntegral $ MP.rows imgA
        w = fromIntegral $ MP.cols imgA
        numElems = h * w * 4  -- RGBA
        flatA = MP.toList imgA
        flatB = MP.toList imgB
    ctx <- getContext []  -- get GPU context
    liftIO $ do
        allocaArray numElems $ \outPtr ->
          withArray (map realToFrac flatA) $ \ptrA ->
          withArray (map realToFrac flatB) $ \ptrB -> do
              ret <- ffiFunc (castPtr ctx) outPtr ptrA ptrB h w
              if ret /= 0
                then error $ "Futhark kernel failed with code " ++ show ret
                else do
                    resultList <- peekArray numElems outPtr
                    pure $ MP.fromListMP (MP.Z MP.:. fromIntegral h MP.:. fromIntegral w) (map realToFrac resultList)

--------------------------------------------------------------------------------
-- | Generic helper for one-input kernels
--------------------------------------------------------------------------------
withOneImage :: (Ptr () -> Ptr CFloat -> CInt -> CInt -> IO CInt)
             -> MP.Image MP.RGBA Float
             -> FutIO (MP.Image MP.RGBA Float)
withOneImage ffiFunc img = do
    let h = fromIntegral $ MP.rows img
        w = fromIntegral $ MP.cols img
        numElems = h * w * 4
        flat = MP.toList img
    ctx <- getContext []
    liftIO $ allocaArray numElems $ \outPtr ->
      withArray (map realToFrac flat) $ \inPtr -> do
          ret <- ffiFunc (castPtr ctx) outPtr inPtr h w
          if ret /= 0
            then error $ "Futhark kernel failed with code " ++ show ret
            else do
                resultList <- peekArray numElems outPtr
                pure $ MP.fromListMP (MP.Z MP.:. fromIntegral h MP.:. fromIntegral w) (map realToFrac resultList)

--------------------------------------------------------------------------------
-- | FFI imports (replace with your actual Futhark C library)
--------------------------------------------------------------------------------
foreign import ccall "diff_diffNoAlphaImages"
    c_diffNoAlphaImages :: Ptr () -> Ptr CFloat -> Ptr CFloat -> CInt -> CInt -> IO CInt

foreign import ccall "diff_diffImages"
    c_diffImages :: Ptr () -> Ptr CFloat -> Ptr CFloat -> CInt -> CInt -> IO CInt

foreign import ccall "blur_blurImage"
    c_blurImage :: Ptr () -> Ptr CFloat -> CInt -> CInt -> IO CInt

foreign import ccall "resize_resizeImage"
    c_resizeImage :: Ptr () -> Ptr CFloat -> CInt -> CInt -> CInt -> CInt -> IO CInt

--------------------------------------------------------------------------------
-- | Public wrappers
--------------------------------------------------------------------------------
diffNoAlphaImages :: MP.Image MP.RGBA Float -> MP.Image MP.RGBA Float -> FutIO (MP.Image MP.RGBA Float)
diffNoAlphaImages = withTwoImages c_diffNoAlphaImages

diffImages :: MP.Image MP.RGBA Float -> MP.Image MP.RGBA Float -> FutIO (MP.Image MP.RGBA Float)
diffImages = withTwoImages c_diffImages

blurImage :: MP.Image MP.RGBA Float -> FutIO (MP.Image MP.RGBA Float)
blurImage = withOneImage c_blurImage

-- resize has extra parameters: new height and width
resizeImage :: MP.Image MP.RGBA Float -> Int -> Int -> FutIO (MP.Image MP.RGBA Float)
resizeImage img newH newW = do
    let h = fromIntegral $ MP.rows img
        w = fromIntegral $ MP.cols img
        numElemsIn = h * w * 4
        numElemsOut = fromIntegral newH * fromIntegral newW * 4
        flat = MP.toList img
    ctx <- getContext []
    liftIO $ allocaArray numElemsOut $ \outPtr ->
      withArray (map realToFrac flat) $ \inPtr -> do
          ret <- c_resizeImage (castPtr ctx) outPtr inPtr h w (fromIntegral newH) (fromIntegral newW)
          if ret /= 0
            then error $ "Futhark resize kernel failed with code " ++ show ret
            else do
                resultList <- peekArray numElemsOut outPtr
                pure $ MP.fromListMP (MP.Z MP.:. newH MP.:. newW) (map realToFrac resultList)
