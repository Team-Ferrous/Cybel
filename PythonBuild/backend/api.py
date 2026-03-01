# api.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
import os
from dotenv import load_dotenv
from pydantic import BaseModel

import main

load_dotenv()

# Global client reference
weaviate_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    global weaviate_client
    
    # ==================== STARTUP ====================
    weaviate_url = os.environ.get("WEAVIATE_URL")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
    
    print("🚀 Starting application...")
    
    # Create Weaviate client
    if os.environ.get("LOCAL_WEAVIATE") == "1":
        weaviate_client = weaviate.connect_to_local(
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=60, query=30, insert=120)
            )
        )
    else:
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
    
    # Initialize the system with the connected client
    main.initialize_system(weaviate_client)
    
    print("✅ Application startup complete")
    
    yield  # Application runs here
    
    # ==================== SHUTDOWN ====================
    print("🛑 Shutting down application...")
    
    # Upload logs to Weaviate before closing
    log_filepath = os.path.join(main.current_dir, 'Logs/chatbot_logs.txt')
    print(f"📤 Uploading logs from {log_filepath} to Weaviate...")
    main.upload_logs_to_weaviate(weaviate_client, log_filepath)
    
    # Close Weaviate client
    if weaviate_client is not None:
        weaviate_client.close()
        print("✅ Weaviate client closed")
    
    print("👋 Shutdown complete")


# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)


# ==================== REQUEST MODELS ====================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


# ==================== ROUTES ====================
@app.get("/")
async def root():
    return {"message": "Cybel Chatbot API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - sends message to the chatbot and returns response.
    """
    try:
        response = main.chat_once(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "weaviate_connected": weaviate_client is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)