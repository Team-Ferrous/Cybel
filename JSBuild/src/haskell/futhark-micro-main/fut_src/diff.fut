-- diff.fut
module DiffNoAlpha where

-- diff two RGBA images ignoring alpha channel
let diffNoAlphaImages (a: [height][width][4]f32) (b: [height][width][4]f32) : [height][width][3]f32 =
  map (\(pa, pb) -> map2 (-) pa pb) (map (take 3) a) (map (take 3) b)
Compile Futhark to a library:

-- Run futhark to compile
--futhark c --library diff.fut
-- produces diff.h, diff.c, diff_stub.h
-- Write Haskell FFI bindings where Lib (In Haskell) can see it:
--foreign import ccall "diff_diffNoAlphaImages"
--c_diffNoAlpha :: Ptr Float -> Ptr Float -> IO (Ptr Float)
