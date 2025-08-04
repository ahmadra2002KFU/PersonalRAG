"""
Temporary mock model loader for testing basic functionality
This allows the app to start without heavy AI dependencies
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MockModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cpu"
        self.mock_loaded = False
        
    def load_qwen3_model(self):
        """Mock load Qwen3-1.7B model"""
        try:
            logger.info("Mock loading Qwen3-1.7B model...")
            self.models["qwen3-1.7b"] = "mock_qwen3_model"
            self.tokenizers["qwen3-1.7b"] = "mock_qwen3_tokenizer"
            logger.info("Qwen3-1.7B model loaded successfully (mock)")
            return True
        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}")
            return False
    
    def load_medgemma_model(self):
        """Mock load MedGemma-4B-IT-GGUF model"""
        try:
            logger.info("Mock loading MedGemma-4B-IT-GGUF model...")
            self.models["medgemma-4b"] = "mock_medgemma_model"
            self.tokenizers["medgemma-4b"] = "mock_medgemma_tokenizer"
            logger.info("MedGemma-4B-IT-GGUF model loaded successfully (mock)")
            self.mock_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load MedGemma-4B-IT-GGUF model: {e}")
            return False
            
    def generate_response(self, model_name: str, prompt: str, max_length: int = 512) -> str:
        """Generate realistic mock response that mimics actual AI behavior"""
        if model_name not in self.models:
            return f"Model {model_name} not loaded"
        
        try:
            # Extract the user query from the prompt by finding the actual question
            user_query = prompt.split("User:")[-1].split("Assistant:")[0].strip()
            if "query '" in prompt and "'" in prompt:
                # Extract query from RAG context format
                start = prompt.find("query '") + 7
                end = prompt.find("'", start)
                if end > start:
                    user_query = prompt[start:end]
            
            if model_name == "medgemma-4b":
                # Generate contextual medical responses based on query
                if "how many patients" in user_query.lower():
                    return """Based on the current hospital records, here are the patient census numbers:

**Total Hospital Census:**
- **383 inpatients** currently in the hospital (target: 375)
- **28 ICU patients** (target: 25 - currently above capacity)
- **8 Emergency Saudi patients** (target: 6)
- **45 Emergency patients** total

**Key Observations:**
- Hospital occupancy rate: 85.1% (383 of 450 licensed beds)
- ICU is at 112% capacity, indicating high critical care demand
- Emergency department showing increased volume
- 67 beds currently available hospital-wide

**Clinical Context:**
This represents a typical high-occupancy period with particular strain on critical care resources. The ICU exceeding target capacity may require monitoring for potential patient flow issues.

*This information is current as of the latest census update and should be verified with real-time systems for clinical decisions.*"""
                
                elif "diabetes" in user_query.lower() and "symptom" in user_query.lower():
                    return """**Common symptoms of diabetes include:**

**Type 1 & Type 2 Diabetes:**
- Increased thirst (polydipsia) and frequent urination (polyuria)
- Unexplained weight loss
- Increased hunger (polyphagia)
- Fatigue and weakness
- Blurred vision
- Slow-healing cuts and wounds
- Frequent infections

**Type 1 specific:**
- Rapid onset of symptoms (days to weeks)
- Diabetic ketoacidosis (DKA) - fruity breath odor, nausea, vomiting

**Type 2 specific:**
- Gradual onset (months to years)
- Darkened skin patches (acanthosis nigricans)
- Tingling or numbness in hands/feet

**Urgent signs requiring immediate medical attention:**
- Severe dehydration, persistent vomiting, difficulty breathing, confusion

*Always consult healthcare professionals for proper diagnosis and treatment. This information is for educational purposes only.*"""
                
                else:
                    # General medical response
                    return f"""Based on the available hospital data and medical knowledge, regarding "{user_query}":

I can provide evidence-based medical information from our hospital records and established clinical guidelines. The specific details would depend on the clinical context and available data.

For accurate medical advice specific to your situation, please consult with qualified healthcare professionals who can assess your individual circumstances.

*This response is generated from the MedGemma-4B-IT medical AI model and hospital database integration.*"""
            
            else:
                # General model response
                return f"""Based on the hospital database query "{user_query}":

I can help you find information from our hospital records, including patient data, equipment status, operational metrics, and department information. The system has access to comprehensive hospital operational data.

Would you like me to search for specific information or provide more details about a particular aspect?"""
            
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
            "cuda_available": False,
            "gpu_memory": None,
            "mock_mode": True,
            "status": "Mock models loaded for testing"
        }

# Global model manager instance
model_manager = MockModelManager()