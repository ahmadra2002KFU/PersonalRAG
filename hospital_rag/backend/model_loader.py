"""
Ollama Model Loading for Hospital RAG System
Uses Ollama to manage and run the gemma3:4b-it-q4_K_M model
"""

import requests
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.loaded_models = set()
        self.default_model = "gemma3:4b-it-q4_K_M"
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False
    
    def load_qwen3_model(self):
        """Load Qwen3 model via Ollama (placeholder - using gemma3 as primary)"""
        # Since we're using gemma3:4b-it-q4_K_M as primary, this is a placeholder
        logger.info("Using gemma3:4b-it-q4_K_M as primary model instead of Qwen3")
        return self.load_medgemma_model()
    
    def load_medgemma_model(self):
        """Load gemma3:4b-it-q4_K_M model via Ollama"""
        try:
            if not self.check_ollama_connection():
                logger.error("Ollama is not running. Please start Ollama first.")
                return False
            
            # Check if model is already available
            model_name = self.default_model
            
            # Try to pull the model if it's not available
            try:
                pull_response = requests.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": model_name},
                    timeout=300  # 5 minutes timeout for model download
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"Model {model_name} pulled successfully")
                else:
                    logger.warning(f"Model pull returned status {pull_response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning("Model pull timed out, but model might already be available")
            except Exception as e:
                logger.warning(f"Model pull failed: {e}, attempting to use existing model")
            
            # Test if model works
            test_response = self.generate_response("medgemma-4b", "Hello")
            if test_response and "error" not in test_response.lower():
                self.loaded_models.add("medgemma-4b")
                logger.info(f"Model {model_name} loaded and tested successfully")
                return True
            else:
                logger.error(f"Model {model_name} test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model via Ollama: {e}")
            return False
    
    def generate_response(self, model_name: str, prompt: str, max_length: int = 512) -> str:
        """Generate response using Ollama"""
        try:
            if not self.check_ollama_connection():
                return "Ollama connection failed. Please ensure Ollama is running."
            
            # Create medical context prompt for medical model
            if model_name == "medgemma-4b":
                system_prompt = "You are a medical AI assistant. Provide accurate, evidence-based medical information. Always include appropriate disclaimers."
                full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = prompt
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.default_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": max_length
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"Error: Ollama API returned status {response.status_code}"
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded in Ollama"""
        return model_name in self.loaded_models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        try:
            if self.check_ollama_connection():
                response = requests.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model["name"] for model in models_data.get("models", [])]
                else:
                    available_models = []
            else:
                available_models = []
                
            return {
                "ollama_url": self.ollama_url,
                "ollama_connected": self.check_ollama_connection(),
                "loaded_models": list(self.loaded_models),
                "available_models": available_models,
                "default_model": self.default_model
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "ollama_url": self.ollama_url,
                "ollama_connected": False,
                "loaded_models": [],
                "available_models": [],
                "default_model": self.default_model,
                "error": str(e)
            }

# Global model manager instance - will be initialized with settings in main.py
model_manager = ModelManager()

def initialize_model_manager(ollama_url: str = "http://localhost:11434"):
    """Initialize the model manager with configuration"""
    global model_manager
    model_manager = ModelManager(ollama_url)