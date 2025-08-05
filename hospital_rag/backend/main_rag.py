"""
Real Hospital RAG System FastAPI Backend
Proper RAG implementation with embeddings, vector search, and Ollama
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from pathlib import Path
import sys
import os
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, MEDICAL_DISCLAIMERS
from backend.rag_engine import HospitalRAGEngine
from backend.ollama_client import OllamaClient

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital RAG System",
    version="2.0.0",
    description="Real RAG system with embeddings and vector search"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    model: str = "llama3.2:3b"
    include_sources: bool = True
    max_sources: int = 5

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    model_used: str
    confidence: float
    disclaimer: str
    processing_time: float

class ModelInfo(BaseModel):
    name: str
    display_name: str
    status: str
    size: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    max_results: int = 10
    filter_type: Optional[str] = None

# Global variables
rag_engine: Optional[HospitalRAGEngine] = None
ollama_client: Optional[OllamaClient] = None
available_models: List[Dict[str, Any]] = []

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_engine, ollama_client, available_models
    
    logger.info("Initializing Hospital RAG System...")
    
    # Initialize RAG engine
    data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
    rag_engine = HospitalRAGEngine(data_dir)
    
    # Initialize database
    rag_engine.initialize_database()
    
    # Initialize Ollama client
    ollama_client = OllamaClient()
    
    # Get available models
    try:
        available_models = await ollama_client.list_models()
        logger.info(f"Found {len(available_models)} available models")
        for model in available_models:
            logger.info(f"  - {model.get('name')}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        available_models = []
    
    # Ensure we have at least one model
    if not available_models:
        logger.warning("No models found. Attempting to pull llama3.2:3b...")
        try:
            result = await ollama_client.pull_model("llama3.2:3b")
            if result.get("success"):
                available_models = await ollama_client.list_models()
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
    
    logger.info("RAG System initialization complete!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Hospital RAG System is running",
        "version": "2.0.0",
        "status": "healthy",
        "database_stats": rag_engine.get_database_stats() if rag_engine else None
    }

@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models"""
    try:
        models = []
        for model in available_models:
            models.append({
                "name": model.get("name"),
                "display_name": model.get("name"),
                "status": "ready",
                "size": model.get("size", "Unknown")
            })
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/stats")
async def get_database_stats():
    """Get RAG database statistics"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        stats = rag_engine.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=List[Dict[str, Any]])
async def semantic_search(query: SearchQuery):
    """Perform semantic search on hospital data"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Build filter if specified
        filter_dict = None
        if query.filter_type:
            filter_dict = {"type": query.filter_type}
        
        # Perform search
        results = rag_engine.semantic_search(
            query.query,
            n_results=query.max_results,
            filter_dict=filter_dict
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_rag(message: ChatMessage):
    """Main RAG chat endpoint"""
    start_time = time.time()
    
    try:
        if not rag_engine or not ollama_client:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Validate model
        model_names = [m.get("name") for m in available_models]
        if message.model not in model_names:
            # Use first available model as fallback
            if model_names:
                message.model = model_names[0]
                logger.warning(f"Model not found, using fallback: {message.model}")
            else:
                raise HTTPException(status_code=400, detail="No models available")
        
        # Perform semantic search to get relevant context
        relevant_docs = rag_engine.semantic_search(
            message.message,
            n_results=message.max_sources
        )
        
        # Generate response using Ollama
        ai_response = await ollama_client.generate_response(
            model=message.model,
            prompt=message.message,
            context_docs=relevant_docs
        )
        
        # Calculate confidence based on search results and AI response
        confidence = 0.1  # Default low confidence
        if relevant_docs:
            avg_similarity = sum(doc["similarity_score"] for doc in relevant_docs) / len(relevant_docs)
            confidence = min(0.95, avg_similarity * 0.9)  # Cap at 95%
        
        if ai_response.get("error"):
            confidence = 0.1
        
        processing_time = time.time() - start_time
        
        # Format sources for response
        sources = []
        if message.include_sources:
            for doc in relevant_docs:
                sources.append({
                    "id": doc["id"],
                    "content": doc["document"][:300] + "..." if len(doc["document"]) > 300 else doc["document"],
                    "metadata": doc["metadata"],
                    "similarity_score": doc["similarity_score"]
                })
        
        return ChatResponse(
            response=ai_response.get("response", "Sorry, I couldn't generate a response."),
            sources=sources,
            model_used=message.model,
            confidence=confidence,
            disclaimer=MEDICAL_DISCLAIMERS["general"],
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            chat_message = ChatMessage(**message_data)
            
            # Send typing indicator
            await manager.send_personal_message(
                json.dumps({"type": "typing", "status": True}), 
                websocket
            )
            
            # Process the message (same logic as REST endpoint)
            try:
                relevant_docs = rag_engine.semantic_search(
                    chat_message.message,
                    n_results=chat_message.max_sources
                )
                
                ai_response = await ollama_client.generate_response(
                    model=chat_message.model,
                    prompt=chat_message.message,
                    context_docs=relevant_docs
                )
                
                # Send response
                await manager.send_personal_message(
                    json.dumps({
                        "type": "message",
                        "response": ai_response.get("response", "Sorry, I couldn't generate a response."),
                        "sources": [{"id": doc["id"], "content": doc["document"][:200] + "...", "similarity": doc["similarity_score"]} for doc in relevant_docs[:3]],
                        "model": chat_message.model,
                        "disclaimer": MEDICAL_DISCLAIMERS["general"]
                    }),
                    websocket
                )
                
            except Exception as e:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": f"Error processing request: {str(e)}"
                    }),
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/reinitialize")
async def reinitialize_database():
    """Reinitialize the RAG database (useful for data updates)"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        logger.info("Reinitializing RAG database...")
        rag_engine.initialize_database()
        
        stats = rag_engine.get_database_stats()
        return {
            "message": "Database reinitialized successfully",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error reinitializing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (frontend)
try:
    static_path = Path(__file__).parent.parent / "frontend"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        
        @app.get("/chat")
        async def serve_chat_interface():
            return FileResponse(str(static_path / "index.html"))
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_rag:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )