module FRPC.Manager where
    -- concurrency & threading
    import Control.Concurrent       (ThreadId, forkOS, MVar, newMVar, readMVar, writeMVar)
    import Control.Concurrent.Chan  (Chan, newChan, readChan, writeChan)

    -- monad transformers & lifting IO
    import Control.Monad.IO.Class   (liftIO)
    import Control.Monad.Trans      (MonadTrans)

    -- identity monad
    import Data.Functor.Identity    (Identity(..))

    -- strictness
    import GHC.Exts                 (inline)
    import Control.DeepSeq          (NFData(..), deepseq)
    import Prelude                  hiding (read)  -- optional if you redefine read

    -- your Futhark binding
    import Futhark                  (getContext, runFutTIn, FutIO, toFuthark, fromFuthark)

    -- your image library
    import qualified Massiv.Array as MP  -- MP.Image, MP.RGBA

    -- exceptions if you want to handle them
    import Control.Exception        (SomeException, try, catch)

    data GPUCommand
    = DiffImages {imgA    :: MP.Image MP.RGBA Float, imgB :: MP.Image MP.RGBA Float}
    | ImageResp  {imgResp :: MP.Image MP.RGBA Float}
    | GPUExit
    deriving (Eq, Show)


    data FutHandle = FutHandle {
    threadID :: ThreadId,
    procSem  :: MVar (Chan GPUCommand) 
    } deriving(Eq, Show)

    newtype FutT c  = FutT c m a (FutT m a)
    newtype FutT     = FutT m a 

    type MFutT m a = FutT m a
    type MFut      = FutT Identity
    type MFutIO    = FutT IO


    futharkProcess :: Chan GPUCommand -> IO ()
    futharkProcess chan = do 
        context <- getContext []
        runFutTIn context $ do 
            let {doFutharkProcess = do
                    command <- liftIO $ readChan chan
                    case command of
                        GPUExit -> return () 
                        _ -> do 
                            result <- handleGPUCommand command
                            liftIO $ writeChan chan result --writeChan
                            doFutharkProcess
                }
            doFutharkProcess


    handleGPUCommand :: GPUCommand -> FutIO GPUCommand
    handleGPUCommand command = do 
        case command of
            DiffImages a b -> do
                futA <- toFuthark $ unImage a
                futB <- toFuthark $ unImage b
                
                !futOutput <- E.diffNoAlphaImages futA futB --E.diffImages futA futB
                --run actual futhark diff operation
                -- return result in ImageResp
                result <- fromFuthark futOutput
                return $ ImageResp $ Image result


    createFutharkHandle :: IO (FutHandle)
    createFutharkHandle = do
        chan <- newChan
        mvar <- newMVar chan
        tid  <- forkOS $ futharkProcess chan
        return $ FutHandle tid mvar