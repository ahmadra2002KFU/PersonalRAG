"""
Hugging Face Client for Hospital RAG System
Replaces Ollama with Qwen2.5-0.5B-Instruct model from Hugging Face
"""

import logging
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """Client for Hugging Face transformers models"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing HuggingFace client with device: {self.device}")
    
    async def initialize_model(self) -> bool:
        """Initialize the Qwen model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"✅ Model {self.model_name} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_name}: {e}")
            return False
    
    async def generate_response(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate response using the loaded model"""
        
        if not self.pipeline:
            return "Error: Model not initialized. Please initialize the model first."
        
        try:
            # Format prompt for Qwen
            formatted_prompt = f"<|im_start|>system\nYou are a helpful medical assistant for a hospital information system. Provide accurate, professional responses based on the given information.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response
            result = self.pipeline(
                formatted_prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            
            # Clean up the response
            response = generated_text.strip()
            if response.endswith("<|im_end|>"):
                response = response[:-10].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models (for compatibility)"""
        return [
            {
                "name": self.model_name,
                "size": "0.5B parameters",
                "status": "loaded" if self.model else "not_loaded"
            }
        ]
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull/load model (for compatibility)"""
        if model_name == self.model_name:
            success = await self.initialize_model()
            return {
                "status": "success" if success else "error",
                "model": model_name
            }
        else:
            return {
                "status": "error",
                "message": f"Model {model_name} not supported. Only {self.model_name} is available."
            }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.pipeline is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.is_model_loaded(),
            "parameters": "0.5B",
            "type": "Qwen2.5 Instruct"
        }

# Global client instance
hf_client = HuggingFaceClient()

async def generate_medical_response(prompt: str, model: str = "qwen") -> str:
    """Generate medical response using Hugging Face model"""
    if not hf_client.is_model_loaded():
        await hf_client.initialize_model()
    
    return await hf_client.generate_response(prompt)

# Test function
async def test_huggingface_client():
    """Test the Hugging Face client"""
    print("Testing Hugging Face Client...")
    
    # Initialize model
    success = await hf_client.initialize_model()
    if not success:
        print("❌ Failed to initialize model")
        return
    
    # Test generation
    test_prompt = "What is a hospital?"
    response = await hf_client.generate_response(test_prompt)
    print(f"Test prompt: {test_prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_huggingface_client())