module Main where

import FRPC

main :: IO ()
main = do
    handle <- createFutharkHandle
    mgr <- initSessionManager handle
    startWorker mgr
    putStrLn "Futhark Micro server running on port 8080..."
    startServer mgr
