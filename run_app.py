#!/usr/bin/env python3
"""
Simple startup script for Hospital RAG System
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("Starting Hospital RAG System...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    
    # Try importing required modules
    print("\nImporting modules...")
    import uvicorn
    print("✓ uvicorn imported")
    
    from hospital_rag.backend.main import app
    print("✓ FastAPI app imported")
    
    # Start the server
    print("\nStarting server on http://0.0.0.0:8000")
    uvicorn.run(
        "hospital_rag.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid multiprocessing issues
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying to install missing dependencies...")
    os.system("pip install fastapi uvicorn pydantic-settings")
    print("Please run the script again after installation.")
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()