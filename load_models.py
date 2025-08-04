#!/usr/bin/env python3
"""
Script to load AI models for the Hospital RAG System
Run this to download and load the actual AI models
"""

import sys
import subprocess
import requests
import json
from pathlib import Path

def install_dependencies():
    """Install required dependencies for model loading"""
    print("Installing model dependencies...")
    dependencies = [
        "torch",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
    
    print("Dependencies installed successfully!")

def check_system_requirements():
    """Check if system has enough resources"""
    import torch
    
    print("\n=== System Requirements Check ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
            
        if memory_gb < 8:
            print("⚠️  Warning: GPU has less than 8GB memory. Models will use 4-bit quantization.")
    else:
        print("⚠️  Warning: No CUDA GPU detected. Models will run on CPU (very slow).")
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System RAM: {ram_gb:.1f}GB")
        
        if ram_gb < 16:
            print("⚠️  Warning: System has less than 16GB RAM. Consider using smaller models.")
    except ImportError:
        print("Install 'psutil' to check RAM: pip install psutil")

def load_model_via_api(model_name):
    """Load model via the hospital RAG API"""
    try:
        print(f"\nLoading {model_name} via API...")
        response = requests.post(f"http://localhost:8000/api/models/load/{model_name}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Failed to load {model_name}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Hospital RAG server is not running!")
        print("Please start the server first with: python hospital_rag/backend/main.py")
        return False
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return False

def check_server_status():
    """Check if the hospital RAG server is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Hospital RAG Server is running")
            print(f"   Status: {data['status']}")
            print(f"   Documents loaded: {data['documents_loaded']}")
            print(f"   Models available: {data['models_available']}")
            return True
        else:
            return False
    except:
        return False

def get_model_info():
    """Get current model information"""
    try:
        response = requests.get("http://localhost:8000/api/models/info")
        if response.status_code == 200:
            info = response.json()
            print("\n=== Model System Info ===")
            print(f"Device: {info['device']}")
            print(f"CUDA Available: {info['cuda_available']}")
            print(f"Loaded Models: {info['loaded_models']}")
            if info['gpu_memory']:
                print(f"GPU Memory: {info['gpu_memory'] / (1024**3):.1f}GB")
    except Exception as e:
        print(f"Error getting model info: {e}")

def main():
    """Main function"""
    print("🏥 Hospital RAG System - Model Loader")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_status():
        print("\n❌ Server is not running. Please start it first:")
        print("   cd /mnt/z/Code/AI\\ SIMA-RAG/0.1Ver")
        print("   source hospital_rag_env/bin/activate")
        print("   python hospital_rag/backend/main.py")
        return
    
    # Get current model info
    get_model_info()
    
    print("\n" + "=" * 50)
    print("Model Loading Options:")
    print("1. Install dependencies only")
    print("2. Load Qwen3-1.7B model")
    print("3. Load MedGemma-4B model (or medical alternative)")
    print("4. Load both models")
    print("5. Check system requirements")
    print("6. Exit")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            install_dependencies()
            
        elif choice == "2":
            if load_model_via_api("qwen3-1.7b"):
                print("🎉 Qwen3-1.7B loaded successfully!")
            
        elif choice == "3":
            if load_model_via_api("medgemma-4b"):
                print("🎉 Medical model loaded successfully!")
            
        elif choice == "4":
            print("Loading both models (this may take a while)...")
            success1 = load_model_via_api("qwen3-1.7b")
            success2 = load_model_via_api("medgemma-4b")
            
            if success1 and success2:
                print("🎉 Both models loaded successfully!")
            elif success1 or success2:
                print("⚠️  One model loaded successfully, one failed.")
            else:
                print("❌ Both models failed to load.")
                
        elif choice == "5":
            try:
                install_dependencies()
                check_system_requirements()
            except Exception as e:
                print(f"Error checking requirements: {e}")
                
        elif choice == "6":
            print("👋 Goodbye!")
            
        else:
            print("Invalid choice. Please try again.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()