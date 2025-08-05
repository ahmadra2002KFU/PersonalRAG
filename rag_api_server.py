"""
Complete RAG API Server
Real RAG system with FastAPI endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
import time
from real_rag_system import RealRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Hospital RAG System",
    version="3.0.0",
    description="Real RAG system with vector embeddings and semantic search"
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
class RAGQuery(BaseModel):
    question: str
    top_k: int = 5
    model: str = "qwen"

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    method: str
    processing_time: float
    total_documents_searched: int

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

# Global RAG system
rag_system: Optional[RealRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    logger.info("🏥 Initializing Hospital RAG System...")
    data_dir = Path(__file__).parent
    rag_system = RealRAGSystem(data_dir)
    stats = rag_system.initialize()
    logger.info(f"✅ RAG System ready! {stats}")

@app.get("/")
async def root():
    """Health check and system info"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    return {
        "message": "Hospital RAG System",
        "version": "3.0.0",
        "status": "ready",
        "system_type": "Real RAG with Vector Embeddings",
        "total_documents": len(rag_system.documents),
        "embedding_dimensions": rag_system.embeddings.shape[1] if len(rag_system.embeddings) > 0 else 0
    }

@app.post("/api/rag", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    """Main RAG endpoint - full RAG pipeline"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = time.time()
    
    try:
        logger.info(f"🔍 RAG Query: {query.question}")
        
        # Perform full RAG query
        result = await rag_system.rag_query(
            question=query.question,
            top_k=query.top_k
        )
        
        processing_time = time.time() - start_time
        
        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            method=result["method"],
            processing_time=round(processing_time, 3),
            total_documents_searched=len(rag_system.documents)
        )
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def vector_search(query: SearchQuery):
    """Vector search endpoint - just retrieval"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"🔍 Vector Search: {query.query}")
        
        # Perform vector search
        results = rag_system.vector_search(
            query=query.query,
            top_k=query.top_k
        )
        
        return {
            "query": query.query,
            "results": results,
            "total_found": len(results),
            "search_method": "vector_embeddings"
        }
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Count document types
    doc_types = {}
    for doc in rag_system.documents:
        doc_type = doc["type"]
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    return {
        "system_info": {
            "type": "Real RAG System",
            "embedding_method": "Semantic Medical Concepts",
            "search_method": "Cosine Similarity Vector Search",
            "llm_integration": "Hugging Face Transformers"
        },
        "data_stats": {
            "total_documents": len(rag_system.documents),
            "document_types": doc_types,
            "embedding_dimensions": rag_system.embeddings.shape[1] if len(rag_system.embeddings) > 0 else 0
        },
        "capabilities": [
            "Semantic vector search",
            "Medical concept understanding",
            "LLM-powered response generation",
            "Real-time query processing"
        ]
    }

# Test endpoints for validation
@app.get("/api/test/saudi-patients")
async def test_saudi_patients():
    """Test endpoint for Saudi patient count"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Use vector search to find Saudi patients
    results = rag_system.vector_search("Saudi Arabian nationality patients", top_k=100)
    
    saudi_patients = [
        doc for doc in results 
        if doc["type"] == "patient" and "saudi arabian" in doc["text"].lower()
    ]
    
    return {
        "query": "Saudi Arabian patients",
        "method": "vector_search",
        "total_documents_searched": len(results),
        "saudi_patients_found": len(saudi_patients),
        "sample_patients": [
            {
                "id": doc["id"],
                "similarity": doc["similarity"],
                "text": doc["text"][:100] + "..."
            }
            for doc in saudi_patients[:5]
        ]
    }

@app.get("/api/test/orthopedic-doctors")
async def test_orthopedic_doctors():
    """Test endpoint for orthopedic specialists"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    results = rag_system.vector_search("orthopedic bone specialist doctor", top_k=20)
    
    doctors = [
        doc for doc in results 
        if doc["type"] == "doctor"
    ]
    
    return {
        "query": "Orthopedic specialists",
        "method": "vector_search", 
        "total_documents_searched": len(results),
        "doctors_found": len(doctors),
        "doctors": [
            {
                "id": doc["id"],
                "similarity": doc["similarity"],
                "text": doc["text"]
            }
            for doc in doctors[:3]
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🏥 Starting Real Hospital RAG System...")
    print("=" * 60)
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔍 RAG endpoint: http://localhost:8000/api/rag")
    print("🔎 Search endpoint: http://localhost:8000/api/search")
    print("📊 Stats: http://localhost:8000/api/stats")
    print("🧪 Test Saudi: http://localhost:8000/api/test/saudi-patients")
    print("🧪 Test Doctors: http://localhost:8000/api/test/orthopedic-doctors")
    print("=" * 60)
    
    uvicorn.run(
        "rag_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )