"""
Ollama Client for Real AI Model Integration
Handles communication with Ollama for actual LLM responses
"""

import requests
import json
import logging
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
                    else:
                        logger.error(f"Failed to list models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def generate_response(self, 
                               model: str, 
                               prompt: str, 
                               context_docs: List[Dict[str, Any]] = None,
                               temperature: float = 0.1,
                               max_tokens: int = 2000) -> Dict[str, Any]:
        """Generate response using Ollama model"""
        
        # Build the full prompt with context
        full_prompt = self._build_rag_prompt(prompt, context_docs)
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "response": data.get("response", "").strip(),
                            "model": model,
                            "done": data.get("done", False),
                            "total_duration": data.get("total_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "eval_count": data.get("eval_count", 0)
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error {response.status}: {error_text}")
                        return {
                            "response": f"Error: Failed to generate response (Status: {response.status})",
                            "model": model,
                            "error": True
                        }
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {
                "response": f"Error: Failed to connect to AI model - {str(e)}",
                "model": model,
                "error": True
            }
    
    def _build_rag_prompt(self, query: str, context_docs: List[Dict[str, Any]] = None) -> str:
        """Build RAG prompt with context documents"""
        
        if not context_docs:
            return f"""You are a helpful medical AI assistant for a hospital system. Please answer the following question accurately and concisely.

Question: {query}

Answer:"""
        
        # Build context from documents
        context = ""
        for i, doc in enumerate(context_docs[:5], 1):  # Limit to top 5 docs
            context += f"\n--- Document {i} ---\n{doc.get('document', '')}\n"
        
        prompt = f"""You are a helpful medical AI assistant for a hospital system. Use the provided hospital data to answer questions accurately. Always base your answers on the actual data provided.

HOSPITAL DATA:
{context}

Question: {query}

Instructions:
- Use only the information provided in the hospital data above
- If asking about counts or numbers, count carefully from the data
- Be specific and include relevant details from the data
- If the data doesn't contain enough information, say so clearly
- For medical information, always include appropriate disclaimers

Answer:"""
        
        return prompt
    
    async def check_model_availability(self, model: str) -> bool:
        """Check if a specific model is available"""
        models = await self.list_models()
        model_names = [m.get("name", "") for m in models]
        return model in model_names
    
    async def pull_model(self, model: str) -> Dict[str, Any]:
        """Pull/download a model if not available"""
        payload = {"name": model}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload
                ) as response:
                    if response.status == 200:
                        return {"success": True, "message": f"Model {model} pulled successfully"}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": error_text}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global client instance
ollama_client = OllamaClient()

async def get_ai_response(query: str, 
                         context_docs: List[Dict[str, Any]] = None, 
                         model: str = "llama3.2:3b") -> Dict[str, Any]:
    """Convenience function to get AI response"""
    return await ollama_client.generate_response(
        model=model,
        prompt=query,
        context_docs=context_docs
    )

if __name__ == "__main__":
    # Test the client
    async def test_client():
        client = OllamaClient()
        
        # List models
        models = await client.list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model.get('name')}")
        
        # Test generation
        if models:
            model_name = models[0].get("name")
            response = await client.generate_response(
                model=model_name,
                prompt="What is 2+2?",
                context_docs=None
            )
            print(f"\nTest response from {model_name}:")
            print(response.get("response"))
    
    asyncio.run(test_client())