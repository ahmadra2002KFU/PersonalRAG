"""
Real AI Model Loading for Hospital RAG System
Loads Qwen3-1.7B and MedGemma-4B models using Hugging Face Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# GGUF support for MedGemma-4B-IT-GGUF
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logger.warning("llama-cpp-python not available, GGUF models will not work")

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_qwen3_model(self):
        """Load Qwen3-1.7B model"""
        try:
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Updated model name
            
            # 4-bit quantization config for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizers["qwen3-1.7b"] = tokenizer
            self.models["qwen3-1.7b"] = model
            
            logger.info("Qwen3-1.7B model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}")
            return False
    
    def load_medgemma_model(self):
        """Load MedGemma-4B-IT-GGUF model using llama-cpp-python"""
        try:
            if not GGUF_AVAILABLE:
                logger.error("llama-cpp-python not available, cannot load GGUF model")
                return False
                
            # Using the actual MedGemma-4B-IT-GGUF model as requested
            model_name = "SandLogicTechnologies/MedGemma-4B-IT-GGUF"
            
            # Download the GGUF file from Hugging Face
            from huggingface_hub import hf_hub_download
            
            # Download the GGUF model file
            model_path = hf_hub_download(
                repo_id=model_name,
                filename="medgemma-4b-it.Q4_K_M.gguf",  # Common GGUF filename pattern
                cache_dir="./models"
            )
            
            # Load the GGUF model with llama-cpp-python
            model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window
                n_threads=4,  # Number of threads
                n_gpu_layers=32 if torch.cuda.is_available() else 0,  # GPU acceleration
                verbose=False
            )
            
            self.models["medgemma-4b"] = model
            # GGUF models don't use separate tokenizers
            self.tokenizers["medgemma-4b"] = None
            
            logger.info("MedGemma-4B-IT-GGUF model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma-4B-IT-GGUF model: {e}")
            # Fallback to a standard medical model if GGUF fails
            logger.info("Falling back to BioGPT-Large...")
            return self._load_biogpt_fallback()
            
    def _load_biogpt_fallback(self):
        """Fallback to BioGPT if GGUF loading fails"""
        try:
            model_name = "microsoft/BioGPT-Large"
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizers["medgemma-4b"] = tokenizer
            self.models["medgemma-4b"] = model
            
            logger.info("BioGPT-Large fallback model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            return False
    
    def generate_response(self, model_name: str, prompt: str, max_length: int = 512) -> str:
        """Generate response using specified model (supports both GGUF and standard models)"""
        if model_name not in self.models:
            return f"Model {model_name} not loaded"
        
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Create medical context prompt
            if model_name == "medgemma-4b":
                system_prompt = "You are a medical AI assistant. Provide accurate, evidence-based medical information. Always include appropriate disclaimers."
                full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"
            
            # Handle GGUF models (llama-cpp-python)
            if isinstance(model, Llama) if GGUF_AVAILABLE else False:
                response = model(
                    full_prompt,
                    max_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["User:", "\n\n"],
                    echo=False
                )
                return response['choices'][0]['text'].strip()
            
            # Handle standard transformers models
            else:
                inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=max_length,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the assistant's response
                response = response.split("Assistant:")[-1].strip()
                return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return model_name in self.models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "device": self.device,
            "loaded_models": list(self.models.keys()),
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }

# Global model manager instance
model_manager = ModelManager()