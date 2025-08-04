"""
Hospital RAG System Configuration
Following medical-grade standards and security practices
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with medical compliance"""
    
    # Application Info
    APP_NAME: str = "Hospital RAG System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Ollama Configuration
    OLLAMA_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "medgemma-4b"  # Maps to gemma3:4b-it-q4_K_M in Ollama
    AVAILABLE_MODELS: List[str] = ["medgemma-4b"]  # Using Ollama gemma3:4b-it-q4_K_M
    
    # Ollama Model Configuration
    OLLAMA_MODEL_NAME: str = "gemma3:4b-it-q4_K_M"  # Actual Ollama model name
    
    # RAG Configuration - Updated to use RAGFlow-compatible embedding models
    VECTOR_DB_TYPE: str = "chromadb"  # chromadb, qdrant, or faiss
    VECTOR_DB_PATH: str = "./data/vectordb"
    # Using BAAI BGE models which are supported by RAGFlow and optimized for medical content
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"  # Primary embedding model from RAGFlow
    MEDICAL_EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"  # Alternative medical embedding model
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_PER_QUERY: int = 5
    
    # Hospital Data Configuration
    HOSPITAL_DATA_DIR: Path = Path("./")
    PATIENT_DATA_FILE: str = "hospital_patients.csv"
    EQUIPMENT_DATA_FILE: str = "hospital_equipment.csv"
    METRICS_DATA_FILE: str = "hospital_operational_metrics.csv"
    
    # Medical Safety Configuration
    ENABLE_MEDICAL_DISCLAIMER: bool = True
    REQUIRE_SOURCE_ATTRIBUTION: bool = True
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_RESPONSE_LENGTH: int = 2000
    
    # Security Configuration
    ENABLE_RATE_LIMITING: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 60
    ENABLE_CORS: bool = True
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "hospital_rag.log"
    ENABLE_MEDICAL_AUDIT_LOG: bool = True
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 10
    MODEL_CACHE_SIZE: int = 2
    ENABLE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.8
    
    class Config:
        env_prefix = "HOSPITAL_RAG_"
        case_sensitive = True

# Medical-specific constants
MEDICAL_DISCLAIMERS = {
    "general": "This information is for educational purposes only and should not replace professional medical advice.",
    "emergency": "For medical emergencies, please contact emergency services immediately.",
    "diagnostic": "This system provides informational support only. All medical decisions should be made by qualified healthcare professionals."
}

MEDICAL_COMMANDS = {
    "/patient": "Search patient records and information",
    "/equipment": "Check medical equipment status and specifications", 
    "/metrics": "View hospital operational metrics and KPIs",
    "/schedule": "Check appointment schedules and availability",
    "/emergency": "Access emergency protocols and procedures",
    "/inventory": "Check department inventory and supplies",
    "/staff": "View staff information and schedules",
    "/dashboard": "Access performance dashboard and analytics"
}

# Color scheme following style.json
UI_COLORS = {
    "primary": {
        "50": "#eff6ff",
        "500": "#3b82f6", 
        "600": "#2563eb",
        "700": "#1d4ed8"
    },
    "semantic": {
        "success": "#10b981",
        "warning": "#f59e0b", 
        "error": "#ef4444",
        "info": "#3b82f6"
    },
    "medical": {
        "critical": "#ef4444",
        "high": "#f59e0b",
        "medium": "#3b82f6",
        "low": "#10b981"
    }
}

# Department mappings
DEPARTMENT_MAPPINGS = {
    "emergency": ["Emergency", "ER", "Trauma", "Urgent Care"],
    "icu": ["ICU", "Intensive Care", "Critical Care"],
    "surgery": ["Surgery", "OR", "Operating Room", "Surgical"],
    "cardiology": ["Cardiology", "Cardiac", "Heart"],
    "radiology": ["Radiology", "Imaging", "X-Ray", "MRI", "CT"],
    "laboratory": ["Laboratory", "Lab", "Pathology", "Blood Bank"],
    "pharmacy": ["Pharmacy", "Medication", "Drug"],
    "pediatrics": ["Pediatrics", "Pediatric", "Children", "NICU"],
    "obstetrics": ["Obstetrics", "OB", "Maternity", "Labor"],
    "oncology": ["Oncology", "Cancer", "Chemotherapy"]
}

# Initialize settings
settings = Settings()