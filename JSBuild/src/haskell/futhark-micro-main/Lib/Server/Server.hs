{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module FRPC.Server where

import GPU.Manager
import E
import qualified Massiv.Array as MP

import Control.Concurrent.STM
import Control.Concurrent
import Control.Monad.IO.Class
import Control.Monad
import Data.Aeson
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as M
import GHC.Generics
import Servant
import Network.Wai.Handler.Warp (run)

-- -----------------------------
-- Session / Job Management
-- -----------------------------
type SessionId = String

data Session = Session
  { sessionStore :: TVar (Maybe (MP.Image MP.RGBA Float))
  , status       :: TVar String
  }

data SessionManager = SessionManager
  { sessions    :: TVar (Map SessionId Session)
  , promptQueue :: TQueue (SessionId, GPUCommand)
  , futHandle   :: FutHandle
  }

initSessionManager :: FutHandle -> IO SessionManager
initSessionManager handle = do
    sessionsVar <- newTVarIO M.empty
    queueVar    <- newTQueueIO
    pure $ SessionManager sessionsVar queueVar handle

startWorker :: SessionManager -> IO ()
startWorker mgr = void $ forkIO $ forever $ do
    (sid, cmd) <- atomically $ readTQueue (promptQueue mgr)
    session <- atomically $ do
        mp <- readTVar (sessions mgr)
        pure $ mp M.! sid
    result <- handleGPUCommand cmd
    atomically $ do
        writeTVar (sessionStore session) (Just $ case result of
            ImageResp img -> img
            _ -> error "Unexpected result")
        writeTVar (status session) "ready"

-- -----------------------------
-- Servant API
-- -----------------------------
data GPUCommandRequest = GPUCommandRequest
  { inputA :: String -- placeholder, replace with actual image serialization
  , inputB :: String
  } deriving (Generic, Show)
instance FromJSON GPUCommandRequest
instance ToJSON GPUCommandRequest

type API =
       "diff"   :> ReqBody '[JSON] GPUCommandRequest :> Post '[JSON] SessionId
  :<|> "result" :> Capture "id" SessionId :> Get '[JSON] (Maybe String) -- placeholder

startServer :: SessionManager -> IO ()
startServer mgr = run 8080 (serve api (server mgr))

api :: Proxy API
api = Proxy

server :: SessionManager -> Server API
server mgr = submitHandler mgr :<|> getResultHandler mgr

submitHandler :: SessionManager -> GPUCommandRequest -> Handler SessionId
submitHandler mgr req = liftIO $ do
    sid <- generateSessionId
    resultVar <- newTVarIO Nothing
    statusVar <- newTVarIO "pending"
    let session = Session resultVar statusVar
    atomically $ do
        modifyTVar' (sessions mgr) (M.insert sid session)
        writeTQueue (promptQueue mgr) (sid, DiffImages dummyImg dummyImg) -- TODO: convert input strings to MP.Image
    pure sid

getResultHandler :: SessionManager -> SessionId -> Handler (Maybe String)
getResultHandler mgr sid = liftIO $ do
    mp <- readTVarIO (sessions mgr)
    case M.lookup sid mp of
        Nothing -> pure Nothing
        Just session -> do
            st <- readTVarIO (status session)
            case st of
                "ready" -> pure $ Just "image-ready" -- TODO: serialize image
                _       -> pure Nothing

-- Dummy placeholder
dummyImg :: MP.Image MP.RGBA Float
dummyImg = MP.makeArray (MP.Z MP.:. 32 MP.:. 32) (\_ -> MP.RGBA 0 0 0 0)

-- Very simple session ID generator
generateSessionId :: IO String
generateSessionId = show <$> randomIO :: IO String
