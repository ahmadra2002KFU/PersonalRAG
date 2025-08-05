"""
Simple RAG Server
Functional RAG system that works immediately
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
import time
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from hospital_rag.backend.simple_rag import get_rag_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Hospital RAG System",
    version="2.0.0",
    description="Functional RAG system with real AI integration"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    model: str = "llama3.2:3b"
    max_sources: int = 10

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    model_used: str
    confidence: float
    processing_time: float

class SearchQuery(BaseModel):
    query: str
    max_results: int = 10
    filter_type: Optional[str] = None

# Global RAG engine
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_engine
    
    logger.info("🏥 Initializing Hospital RAG System...")
    data_dir = Path(__file__).parent
    rag_engine = get_rag_engine(data_dir)
    logger.info("✅ RAG System ready!")

@app.get("/")
async def root():
    """Health check"""
    return {
        "message": "Hospital RAG System is running", 
        "version": "2.0.0",
        "status": "healthy",
        "stats": rag_engine.get_stats() if rag_engine else None
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return rag_engine.get_stats()

@app.post("/api/search")
async def search(query: SearchQuery):
    """Semantic search endpoint"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        results = rag_engine.semantic_search(
            query.query,
            max_results=query.max_results,
            filter_type=query.filter_type
        )
        
        return {
            "query": query.query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main RAG chat endpoint"""
    start_time = time.time()
    
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        # Perform semantic search
        logger.info(f"🔍 Processing query: {message.message}")
        search_results = rag_engine.semantic_search(
            message.message,
            max_results=message.max_sources
        )
        
        logger.info(f"📊 Found {len(search_results)} relevant documents")
        
        # Generate AI response
        ai_response = await rag_engine.generate_ai_response(
            message.message,
            search_results,
            message.model
        )
        
        # Calculate confidence
        confidence = 0.1
        if search_results:
            avg_relevance = sum(doc.get("relevance_score", 0) for doc in search_results) / len(search_results)
            confidence = min(0.95, avg_relevance / 25)  # Normalize to 0-0.95
        
        if ai_response.get("success"):
            confidence = min(0.95, confidence + 0.2)
        
        processing_time = time.time() - start_time
        
        # Format sources
        sources = []
        for doc in search_results[:5]:  # Top 5 sources
            sources.append({
                "id": doc["id"],
                "type": doc["type"],
                "content": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "metadata": doc["metadata"],
                "relevance_score": doc.get("relevance_score", 0)
            })
        
        return ChatResponse(
            response=ai_response["response"],
            sources=sources,
            model_used=ai_response["model"],
            confidence=confidence,
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("🏥 Starting Hospital RAG System...")
    print("=" * 50)
    print("🌐 Server will be available at: http://localhost:8000")
    print("📊 Stats endpoint: http://localhost:8000/api/stats")
    print("🔍 Search endpoint: http://localhost:8000/api/search")
    print("💬 Chat endpoint: http://localhost:8000/api/chat")
    print("=" * 50)
    
    uvicorn.run(
        "run_rag_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )