"""
Hospital RAG System FastAPI Backend
Medical-grade RAG system with local AI models
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, MEDICAL_DISCLAIMERS, MEDICAL_COMMANDS
from backend.simple_data_processor import SimpleHospitalDataProcessor
try:
    from backend.model_loader import model_manager, initialize_model_manager
except ImportError:
    # Fallback to mock model loader if dependencies missing
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from temp_model_loader import model_manager, initialize_model_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Medical-grade RAG system for hospital operations"
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
    model: str = "medgemma-4b"
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
    description: str
    status: str
    parameters: str

class HospitalQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    query_type: str = "general"

# Global variables for models and data
hospital_documents = []
mock_models = {
    "medgemma-4b": {
        "name": "medgemma-4b", 
        "display_name": "Gemma3-4B Medical",
        "description": "Medical domain specialized model via Ollama (gemma3:4b-it-q4_K_M)",
        "status": "ready",
        "parameters": "4B (4-bit quantized via Ollama)"
    }
}

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global hospital_documents
    
    logger.info("Starting Hospital RAG System...")
    
    # Load processed hospital data
    try:
        data_path = Path("./Database/processed_hospital_data.json")
        if data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                hospital_documents = json.load(f)
            logger.info(f"Loaded {len(hospital_documents)} hospital documents")
        else:
            logger.warning("No processed hospital data found, processing now...")
            processor = SimpleHospitalDataProcessor(Path("./Database"))
            hospital_documents = processor.process_all_data()
            processor.save_processed_data(hospital_documents)
            logger.info(f"Processed and loaded {len(hospital_documents)} hospital documents")
    except Exception as e:
        logger.error(f"Error loading hospital data: {e}")
        hospital_documents = []
    
    # Initialize and load AI models via Ollama
    logger.info("Initializing Ollama model manager...")
    try:
        # Initialize model manager with Ollama URL from settings
        initialize_model_manager(settings.OLLAMA_URL)
        logger.info(f"Model manager initialized with Ollama URL: {settings.OLLAMA_URL}")
        
        # Load gemma3:4b-it-q4_K_M via Ollama
        success = model_manager.load_medgemma_model()
        if success:
            logger.info("Gemma3:4b-it-q4_K_M model loaded successfully via Ollama")
        else:
            logger.warning("Failed to load model via Ollama, will use fallback responses")
    except Exception as e:
        logger.error(f"Error loading AI models via Ollama: {e}")
        logger.info("Falling back to mock responses - ensure Ollama is running")
    
    logger.info("Hospital RAG System started successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Hospital RAG System API", "version": settings.APP_VERSION}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": len(mock_models),
        "documents_loaded": len(hospital_documents),
        "version": settings.APP_VERSION
    }

@app.get("/api/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available AI models"""
    models_info = []
    for model_name, model_data in mock_models.items():
        # Update status based on whether real model is loaded
        status = "loaded" if model_manager.is_model_loaded(model_name) else "mock"
        model_info = model_data.copy()
        model_info["status"] = status
        models_info.append(ModelInfo(**model_info))
    return models_info

@app.post("/api/models/load/{model_name}")
async def load_model(model_name: str):
    """Load a specific AI model via Ollama"""
    try:
        if model_name == "medgemma-4b":
            success = model_manager.load_medgemma_model()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}. Only 'medgemma-4b' is supported.")
        
        if success:
            return {"message": f"Model {model_name} (gemma3:4b-it-q4_K_M) loaded successfully via Ollama", "status": "loaded"}
        else:
            return {"message": f"Failed to load model {model_name}. Ensure Ollama is running and gemma3:4b-it-q4_K_M is available.", "status": "failed"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/info")
async def get_model_system_info():
    """Get system information about models"""
    return model_manager.get_model_info()

@app.get("/api/commands")
async def get_medical_commands():
    """Get available medical commands"""
    return {
        "commands": MEDICAL_COMMANDS,
        "usage": "Type / followed by a command name to get started"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_rag(message: ChatMessage):
    """Main chat endpoint with RAG functionality"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Validate model
        if message.model not in mock_models:
            raise HTTPException(status_code=400, detail=f"Model {message.model} not available")
        
        # Process medical command if present
        if message.message.startswith('/'):
            response = await process_medical_command(message.message, message.model)
        else:
            response = await process_regular_query(message.message, message.model, message.max_sources)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ChatResponse(
            response=response["answer"],
            sources=response["sources"],
            model_used=message.model,
            confidence=response["confidence"],
            disclaimer=response["disclaimer"],
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/hospital")
async def query_hospital_data(query: HospitalQuery):
    """Specialized endpoint for hospital data queries"""
    try:
        # Search hospital documents
        relevant_docs = search_hospital_documents(query.query, query.filters, query.query_type)
        
        return {
            "query": query.query,
            "results": relevant_docs[:10],  # Limit to top 10 results
            "total_found": len(relevant_docs),
            "query_type": query.query_type
        }
        
    except Exception as e:
        logger.error(f"Error in hospital query: {e}")
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
            
            # Process the message
            if chat_message.message.startswith('/'):
                response = await process_medical_command(chat_message.message, chat_message.model)
            else:
                response = await process_regular_query(chat_message.message, chat_message.model, chat_message.max_sources)
            
            # Send response
            await manager.send_personal_message(
                json.dumps({
                    "type": "message",
                    "response": response["answer"],
                    "sources": response["sources"],
                    "model": chat_message.model,
                    "disclaimer": response["disclaimer"]
                }),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Helper functions
async def process_medical_command(command: str, model: str) -> Dict[str, Any]:
    """Process medical commands like /patient, /equipment, etc."""
    command_parts = command.strip().split()
    command_name = command_parts[0].lower()
    
    if command_name == "/patient":
        if len(command_parts) > 1:
            patient_query = " ".join(command_parts[1:])
            return await search_patients(patient_query, model)
        else:
            return {
                "answer": "Please specify patient criteria. Example: /patient Saudi or /patient P0001",
                "sources": [],
                "confidence": 1.0,
                "disclaimer": MEDICAL_DISCLAIMERS["general"]
            }
    
    elif command_name == "/equipment":
        if len(command_parts) > 1:
            equipment_query = " ".join(command_parts[1:])
            return await search_equipment(equipment_query, model)
        else:
            return await get_equipment_overview(model)
    
    elif command_name == "/metrics":
        if len(command_parts) > 1:
            dept = " ".join(command_parts[1:])
            return await get_department_metrics(dept, model)
        else:
            return await get_hospital_metrics_overview(model)
    
    elif command_name == "/emergency":
        return await get_emergency_info(model)
    
    else:
        available_commands = "\n".join([f"{cmd}: {desc}" for cmd, desc in MEDICAL_COMMANDS.items()])
        return {
            "answer": f"Unknown command. Available commands:\n\n{available_commands}",
            "sources": [],
            "confidence": 1.0,
            "disclaimer": MEDICAL_DISCLAIMERS["general"]
        }

async def process_regular_query(query: str, model: str, max_sources: int) -> Dict[str, Any]:
    """Process regular text queries with RAG"""
    # Minimum query length check
    if len(query.strip()) < 3:
        return {
            "answer": "Please provide a more specific query (at least 3 characters). Try asking about patients, equipment, metrics, or use commands like /patient, /equipment, /metrics.",
            "sources": [],
            "confidence": 1.0,
            "disclaimer": MEDICAL_DISCLAIMERS["general"]
        }
    
    # Check if this is a demographic/counting query
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["how many", "count", "number of"]) and any(demo in query_lower for demo in ["female", "male", "patient", "gender"]):
        return await process_demographic_query(query, model)
    
    # Search relevant documents
    relevant_docs = search_hospital_documents(query, None, "general")
    top_docs = relevant_docs[:max_sources]
    
    # Generate response based on model
    if model == "medgemma-4b":
        response = generate_medical_response(query, top_docs)
    else:
        response = generate_general_response(query, top_docs)
    
    return {
        "answer": response,
        "sources": [{"title": doc["title"], "content": doc["content"][:200] + "...", "metadata": doc["metadata"]} for doc in top_docs],
        "confidence": 0.85 if top_docs else 0.1,
        "disclaimer": MEDICAL_DISCLAIMERS["general"]
    }

async def process_demographic_query(query: str, model: str) -> Dict[str, Any]:
    """Process demographic queries like 'How many female patients do we have?'"""
    query_lower = query.lower()
    
    # Get all patient records
    patient_docs = [doc for doc in hospital_documents if doc["type"] == "patient_record"]
    
    # Count by gender
    if "female" in query_lower:
        female_patients = [doc for doc in patient_docs if "Gender: Female" in doc["content"]]
        count = len(female_patients)
        gender = "female"
        sample_patients = female_patients[:5]  # Show first 5 as examples
    elif "male" in query_lower:
        male_patients = [doc for doc in patient_docs if "Gender: Male" in doc["content"]]
        count = len(male_patients)
        gender = "male"
        sample_patients = male_patients[:5]
    else:
        # General patient count
        count = len(patient_docs)
        gender = "total"
        sample_patients = patient_docs[:5]
    
    # Generate response
    if gender == "female":
        response = f"Based on the hospital database, we have **{count} female patients**.\n\n"
    elif gender == "male":
        response = f"Based on the hospital database, we have **{count} male patients**.\n\n"
    else:
        response = f"Based on the hospital database, we have **{count} total patients**.\n\n"
    
    # Add sample patients for context
    if sample_patients and count > 0:
        response += "Sample patients:\n"
        for i, patient in enumerate(sample_patients, 1):
            patient_name = patient["title"].replace("Patient Record: ", "")
            patient_id = patient["metadata"]["patient_id"]
            status = patient["metadata"]["status"]
            response += f"{i}. {patient_name} (ID: {patient_id}, Status: {status})\n"
        
        if count > 5:
            response += f"... and {count - 5} more {gender} patients.\n"
    
    return {
        "answer": response,
        "sources": sample_patients[:3],  # Include top 3 as sources
        "confidence": 1.0,
        "disclaimer": MEDICAL_DISCLAIMERS["general"]
    }

def search_hospital_documents(query: str, filters: Optional[Dict] = None, query_type: str = "general") -> List[Dict[str, Any]]:
    """Search hospital documents using improved text matching"""
    query_lower = query.lower().strip()
    
    # Skip very short queries
    if len(query_lower) < 3:
        return []
    
    results = []
    query_words = query_lower.split()
    
    for doc in hospital_documents:
        score = 0
        title_lower = doc["title"].lower()
        content_lower = doc["content"].lower()
        
        # Exact phrase matching (highest weight)
        if query_lower in title_lower:
            score += 10
        elif query_lower in content_lower:
            score += 8
        
        # Word matching with higher thresholds
        word_matches = 0
        for word in query_words:
            if len(word) >= 3:  # Only consider words of 3+ characters
                if word in title_lower:
                    word_matches += 3
                elif word in content_lower:
                    word_matches += 2
                
                # Metadata matching for meaningful words
                for key, value in doc["metadata"].items():
                    if isinstance(value, str) and word in value.lower():
                        word_matches += 1
        
        score += word_matches
        
        # Apply filters
        if filters:
            for filter_key, filter_value in filters.items():
                if filter_key in doc["metadata"] and doc["metadata"][filter_key] != filter_value:
                    score = 0  # Exclude if doesn't match filter
                    break
        
        # Lower threshold for gender/demographic queries
        min_threshold = 1 if any(demo in query_lower for demo in ["female", "male", "gender", "patient"]) else 3
        
        # Only include results with meaningful scores
        if score >= min_threshold:
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = score
            results.append(doc_with_score)
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results

async def search_patients(query: str, model: str) -> Dict[str, Any]:
    """Search for patients"""
    # Search for patients specifically
    patient_docs = [doc for doc in hospital_documents if doc["type"] == "patient_record"]
    
    results = []
    query_lower = query.lower()
    
    for doc in patient_docs:
        if (query_lower in doc["content"].lower() or 
            query_lower in doc["title"].lower() or
            any(query_lower in str(v).lower() for v in doc["metadata"].values())):
            results.append(doc)
    
    if results:
        response = f"Found {len(results)} patient(s) matching '{query}':\n\n"
        for i, doc in enumerate(results[:5]):  # Limit to 5 results
            response += f"{i+1}. {doc['title']}\n"
            response += f"   Status: {doc['metadata'].get('status', 'Unknown')}\n"
            response += f"   Department: {doc['metadata'].get('department', 'Unknown')}\n\n"
    else:
        response = f"No patients found matching '{query}'. Please check the patient ID or name."
    
    return {
        "answer": response,
        "sources": results[:3],
        "confidence": 0.9 if results else 0.1,
        "disclaimer": MEDICAL_DISCLAIMERS["general"]
    }

async def search_equipment(query: str, model: str) -> Dict[str, Any]:
    """Search for equipment"""
    equipment_docs = [doc for doc in hospital_documents if doc["type"] == "equipment_record"]
    
    results = []
    query_lower = query.lower()
    
    for doc in equipment_docs:
        if (query_lower in doc["content"].lower() or 
            query_lower in doc["title"].lower() or
            query_lower in doc["metadata"].get("department", "").lower()):
            results.append(doc)
    
    if results:
        response = f"Found {len(results)} equipment item(s) matching '{query}':\n\n"
        for i, doc in enumerate(results[:5]):
            response += f"{i+1}. {doc['title']}\n"
            response += f"   Department: {doc['metadata'].get('department', 'Unknown')}\n"
            response += f"   Status: {doc['metadata'].get('status', 'Unknown')}\n"
            response += f"   Location: {doc['metadata'].get('location', 'Unknown')}\n\n"
    else:
        response = f"No equipment found matching '{query}'. Try searching by department or equipment type."
    
    return {
        "answer": response,
        "sources": results[:3],
        "confidence": 0.9 if results else 0.1,
        "disclaimer": MEDICAL_DISCLAIMERS["general"]
    }

async def get_equipment_overview(model: str) -> Dict[str, Any]:
    """Get equipment overview"""
    equipment_docs = [doc for doc in hospital_documents if doc["type"] == "equipment_record"]
    
    # Count by status
    status_counts = {}
    dept_counts = {}
    
    for doc in equipment_docs:
        status = doc["metadata"].get("status", "Unknown")
        dept = doc["metadata"].get("department", "Unknown")
        
        status_counts[status] = status_counts.get(status, 0) + 1
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    response = f"Hospital Equipment Overview ({len(equipment_docs)} total items):\n\n"
    response += "By Status:\n"
    for status, count in sorted(status_counts.items()):
        response += f"  • {status}: {count} items\n"
    
    response += "\nBy Department:\n"
    for dept, count in sorted(dept_counts.items()):
        response += f"  • {dept}: {count} items\n"
    
    return {
        "answer": response,
        "sources": equipment_docs[:3],
        "confidence": 1.0,
        "disclaimer": MEDICAL_DISCLAIMERS["general"]
    }

async def get_hospital_metrics_overview(model: str) -> Dict[str, Any]:
    """Get hospital metrics overview"""
    metric_docs = [doc for doc in hospital_documents if doc["type"] == "operational_metric"]
    
    if not metric_docs:
        return {
            "answer": "No operational metrics data available.",
            "sources": [],
            "confidence": 0.1,
            "disclaimer": MEDICAL_DISCLAIMERS["general"]
        }
    
    # Group by category
    categories = {}
    for doc in metric_docs:
        category = doc["metadata"].get("category", "Unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(doc)
    
    response = f"Hospital Operational Metrics Overview ({len(metric_docs)} total metrics):\n\n"
    
    for category, docs in sorted(categories.items()):
        response += f"{category}: {len(docs)} metrics\n"
        # Show a few key metrics from each category
        for doc in docs[:2]:
            response += f"  • {doc['metadata']['metric_name']}: {doc['metadata']['current_value']} {doc['metadata'].get('unit', '')}\n"
        if len(docs) > 2:
            response += f"  • ... and {len(docs) - 2} more\n"
        response += "\n"
    
    return {
        "answer": response,
        "sources": metric_docs[:5],
        "confidence": 1.0,
        "disclaimer": MEDICAL_DISCLAIMERS["general"]
    }

async def get_emergency_info(model: str) -> Dict[str, Any]:
    """Get emergency department information"""
    # Search for emergency-related documents
    emergency_docs = []
    for doc in hospital_documents:
        if ("emergency" in doc["content"].lower() or 
            "emergency" in doc["title"].lower() or
            doc["metadata"].get("department", "").lower() == "emergency"):
            emergency_docs.append(doc)
    
    response = f"Emergency Department Information:\n\n"
    
    # Find current emergency metrics
    for doc in emergency_docs:
        if doc["type"] == "operational_metric" and "emergency" in doc["metadata"].get("category", "").lower():
            response += f"• {doc['metadata']['metric_name']}: {doc['metadata']['current_value']} {doc['metadata'].get('unit', '')}\n"
    
    response += "\nFor medical emergencies, please contact emergency services immediately at 911 or your local emergency number.\n"
    
    return {
        "answer": response,
        "sources": emergency_docs[:3],
        "confidence": 1.0,
        "disclaimer": MEDICAL_DISCLAIMERS["emergency"]
    }

def generate_medical_response(query: str, docs: List[Dict]) -> str:
    """Generate response using medical model (MedGemma or simulation)"""
    if not docs:
        if model_manager.is_model_loaded("medgemma-4b"):
            prompt = f"Based on general medical knowledge, please provide information about: {query}"
            return model_manager.generate_response("medgemma-4b", prompt)
        else:
            return "I don't have specific information about that in the hospital database. Please consult with medical staff for accurate information."
    
    # Use real model if loaded, otherwise simulate
    if model_manager.is_model_loaded("medgemma-4b"):
        context = "\n".join([doc["content"][:300] for doc in docs[:3]])
        prompt = f"Based on the following hospital records, please provide a medical analysis for the query '{query}':\n\nHospital Records:\n{context}\n\nPlease provide a professional medical response with appropriate disclaimers."
        return model_manager.generate_response("medgemma-4b", prompt)
    else:
        # Simulate medical model response
        response = f"Based on the hospital records, here's what I found:\n\n"
        
        # Add relevant information from documents
        for i, doc in enumerate(docs[:2]):
            response += f"{i+1}. {doc['title']}\n"
            response += f"   {doc['content'][:150]}...\n\n"
        
        response += "Please note: This information is from hospital records and should be verified with medical staff for clinical decisions."
        
        return response

def generate_general_response(query: str, docs: List[Dict]) -> str:
    """Generate response using medical model (fallback to simulation if not available)"""
    if not docs:
        if model_manager.is_model_loaded("medgemma-4b"):
            prompt = f"Please provide general information about: {query}"
            return model_manager.generate_response("medgemma-4b", prompt)
        else:
            return "I couldn't find specific information about that in the hospital database. Could you please rephrase your query or try a different search term?"
    
    # Use real model if loaded, otherwise simulate
    if model_manager.is_model_loaded("medgemma-4b"):
        context = "\n".join([doc["content"][:300] for doc in docs[:3]])
        prompt = f"Based on the following hospital database information, please provide a helpful response to the query '{query}':\n\nDatabase Information:\n{context}\n\nPlease summarize the key information in a clear, helpful way."
        return model_manager.generate_response("medgemma-4b", prompt)
    else:
        # Simulate model response
        response = f"I found the following information in the hospital database:\n\n"
        
        for i, doc in enumerate(docs[:3]):
            response += f"• {doc['title']}\n"
            response += f"  {doc['content'][:200]}...\n\n"
        
        if len(docs) > 3:
            response += f"... and {len(docs) - 3} more related records."
        
        return response

# Serve static files
app.mount("/static", StaticFiles(directory="hospital_rag/frontend"), name="static")

@app.get("/chat")
async def serve_chat_interface():
    """Serve the chat interface"""
    return FileResponse("hospital_rag/frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )