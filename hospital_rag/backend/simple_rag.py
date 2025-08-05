"""
Simple but Functional RAG System
Works without heavy dependencies while providing real RAG functionality
"""

import csv
import json
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

class SimpleRAGEngine:
    """Simple RAG engine that works with basic text search and real AI"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.documents = []
        self.initialized = False
        
    def load_hospital_data(self) -> List[Dict[str, Any]]:
        """Load all hospital CSV files into searchable documents"""
        documents = []
        
        # Load patient data
        try:
            with open(self.data_dir / "hospital_patients.csv", 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for patient in reader:
                    doc_text = f"""
                    Patient ID: {patient['Patient_ID']}
                    Name: {patient['First_Name']} {patient['Last_Name']}
                    Gender: {patient['Gender']}
                    Nationality: {patient.get('Nationality', 'Unknown')}
                    Date of Birth: {patient['Date_of_Birth']}
                    Phone: {patient.get('Phone', 'Unknown')}
                    Address: {patient.get('Address', 'Unknown')}
                    Insurance Type: {patient.get('Insurance_Type', 'Unknown')}
                    Blood Type: {patient.get('Blood_Type', 'Unknown')}
                    Current Status: {patient.get('Current_Status', 'Unknown')}
                    Allergies: {patient.get('Allergies', 'None')}
                    Admission Date: {patient.get('Admission_Date', 'N/A')}
                    Discharge Date: {patient.get('Discharge_Date', 'N/A')}
                    Emergency Contact: {patient.get('Emergency_Contact', 'Unknown')}
                    """.strip()
                    
                    documents.append({
                        "id": f"patient_{patient['Patient_ID']}",
                        "text": doc_text,
                        "type": "patient",
                        "metadata": {
                            "patient_id": patient['Patient_ID'],
                            "name": f"{patient['First_Name']} {patient['Last_Name']}",
                            "gender": patient['Gender'],
                            "nationality": patient.get('Nationality', 'Unknown'),
                            "status": patient.get('Current_Status', 'Unknown'),
                            "source": "hospital_patients.csv"
                        }
                    })
            
            logger.info(f"Loaded {len([d for d in documents if d['type'] == 'patient'])} patient records")
            
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
        
        # Load equipment data
        try:
            with open(self.data_dir / "hospital_equipment.csv", 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for equipment in reader:
                    doc_text = f"""
                    Equipment ID: {equipment['Equipment_ID']}
                    Equipment Name: {equipment['Equipment_Name']}
                    Model: {equipment['Model']}
                    Manufacturer: {equipment['Manufacturer']}
                    Department: {equipment['Department']}
                    Location: {equipment.get('Location', 'Unknown')}
                    Status: {equipment['Status']}
                    Cost: ${equipment.get('Cost', 0)}
                    Last Maintenance: {equipment.get('Last_Maintenance', 'Unknown')}
                    Next Maintenance: {equipment.get('Next_Maintenance', 'Unknown')}
                    Specifications: {equipment.get('Specifications', 'N/A')}
                    """.strip()
                    
                    documents.append({
                        "id": f"equipment_{equipment['Equipment_ID']}",
                        "text": doc_text,
                        "type": "equipment",
                        "metadata": {
                            "equipment_id": equipment['Equipment_ID'],
                            "name": equipment['Equipment_Name'],
                            "department": equipment['Department'],
                            "status": equipment['Status'],
                            "source": "hospital_equipment.csv"
                        }
                    })
            
            logger.info(f"Loaded {len([d for d in documents if d['type'] == 'equipment'])} equipment records")
            
        except Exception as e:
            logger.error(f"Error loading equipment data: {e}")
        
        # Load medical records
        try:
            with open(self.data_dir / "hospital_medical_records.csv", 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for record in reader:
                    doc_text = f"""
                    Medical Record ID: {record['Record_ID']}
                    Patient ID: {record['Patient_ID']}
                    Doctor ID: {record['Doctor_ID']}
                    Date: {record['Date']}
                    Diagnosis: {record['Diagnosis']}
                    Treatment: {record['Treatment']}
                    Medications: {record.get('Medications', 'None')}
                    Vital Signs: {record.get('Vital_Signs', 'Not recorded')}
                    Lab Results: {record.get('Lab_Results', 'N/A')}
                    Clinical Notes: {record.get('Notes', 'N/A')}
                    Follow-up Date: {record.get('Follow_up_Date', 'N/A')}
                    """.strip()
                    
                    documents.append({
                        "id": f"record_{record['Record_ID']}",
                        "text": doc_text,
                        "type": "medical_record",
                        "metadata": {
                            "record_id": record['Record_ID'],
                            "patient_id": record['Patient_ID'],
                            "doctor_id": record['Doctor_ID'],
                            "diagnosis": record['Diagnosis'],
                            "date": record['Date'],
                            "source": "hospital_medical_records.csv"
                        }
                    })
            
            logger.info(f"Loaded {len([d for d in documents if d['type'] == 'medical_record'])} medical records")
            
        except Exception as e:
            logger.error(f"Error loading medical records: {e}")
        
        # Load doctors data
        try:
            with open(self.data_dir / "hospital_doctors.csv", 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for doctor in reader:
                    doc_text = f"""
                    Doctor ID: {doctor['Doctor_ID']}
                    Name: Dr. {doctor['First_Name']} {doctor['Last_Name']}
                    Specialty: {doctor['Specialty']}
                    Department: {doctor['Department']}
                    License Number: {doctor['License_Number']}
                    Phone: {doctor.get('Phone', 'Unknown')}
                    Email: {doctor.get('Email', 'Unknown')}
                    Years of Experience: {doctor.get('Years_Experience', 'Unknown')} years
                    Education: {doctor.get('Education', 'Unknown')}
                    Availability: {doctor.get('Availability', 'Unknown')}
                    Shift Schedule: {doctor.get('Shift_Schedule', 'Unknown')}
                    Salary: ${doctor.get('Salary', 'Unknown')}
                    Employment Status: {doctor.get('Employee_Status', 'Unknown')}
                    """.strip()
                    
                    documents.append({
                        "id": f"doctor_{doctor['Doctor_ID']}",
                        "text": doc_text,
                        "type": "doctor",
                        "metadata": {
                            "doctor_id": doctor['Doctor_ID'],
                            "name": f"Dr. {doctor['First_Name']} {doctor['Last_Name']}",
                            "specialty": doctor['Specialty'],
                            "department": doctor['Department'],
                            "experience": doctor.get('Years_Experience', 'Unknown'),
                            "availability": doctor.get('Availability', 'Unknown'),
                            "source": "hospital_doctors.csv"
                        }
                    })
            
            logger.info(f"Loaded {len([d for d in documents if d['type'] == 'doctor'])} doctor records")
            
        except Exception as e:
            logger.error(f"Error loading doctor data: {e}")
        
        # Load departments data
        try:
            with open(self.data_dir / "hospital_departments.csv", 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for dept in reader:
                    doc_text = f"""
                    Department ID: {dept['Department_ID']}
                    Department Name: {dept['Department_Name']}
                    Head of Department: {dept['Head_of_Department']}
                    Location: {dept['Location']}
                    Floor: {dept['Floor']}
                    Total Beds: {dept['Total_Beds']}
                    Available Beds: {dept['Available_Beds']}
                    Occupied Beds: {int(dept['Total_Beds']) - int(dept['Available_Beds'])}
                    Equipment Count: {dept['Equipment_Count']}
                    Annual Budget: ${dept.get('Annual_Budget', 'Unknown')}
                    Phone Extension: {dept.get('Phone_Extension', 'Unknown')}
                    Emergency Contact: {dept.get('Emergency_Contact', 'Unknown')}
                    """.strip()
                    
                    documents.append({
                        "id": f"department_{dept['Department_ID']}",
                        "text": doc_text,
                        "type": "department",
                        "metadata": {
                            "department_id": dept['Department_ID'],
                            "department_name": dept['Department_Name'],
                            "head_of_department": dept['Head_of_Department'],
                            "total_beds": int(dept['Total_Beds']),
                            "available_beds": int(dept['Available_Beds']),
                            "occupied_beds": int(dept['Total_Beds']) - int(dept['Available_Beds']),
                            "equipment_count": int(dept['Equipment_Count']),
                            "location": dept['Location'],
                            "floor": dept['Floor'],
                            "source": "hospital_departments.csv"
                        }
                    })
            
            logger.info(f"Loaded {len([d for d in documents if d['type'] == 'department'])} department records")
            
        except Exception as e:
            logger.error(f"Error loading department data: {e}")
        
        return documents
    
    def semantic_search(self, query: str, max_results: int = 10, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform intelligent text search (simulates semantic search)"""
        if not self.documents:
            return []
        
        query_lower = query.lower()
        results = []
        
        # Define search patterns for different types of queries
        search_patterns = {
            "saudi": ["saudi", "saudi arabian", "nationality saudi"],
            "gender": ["male", "female", "gender"],
            "patient": ["patient", "patients"],  
            "equipment": ["equipment", "machine", "device"],
            "department": ["department", "ward", "unit"],
            "status": ["status", "condition", "state"]
        }
        
        # Score each document
        for doc in self.documents:
            # Apply type filter if specified
            if filter_type and doc["type"] != filter_type:
                continue
                
            score = 0
            text_lower = doc["text"].lower()
            
            # Exact phrase matching (highest score)
            if query_lower in text_lower:
                score += 20
            
            # Pattern-based scoring
            for pattern, keywords in search_patterns.items():
                if any(keyword in query_lower for keyword in keywords):
                    if any(keyword in text_lower for keyword in keywords):
                        score += 15
            
            # Word-by-word matching
            query_words = [word for word in query_lower.split() if len(word) > 2]
            for word in query_words:
                if word in text_lower:
                    score += 3
                    
                # Check metadata for additional matches
                for key, value in doc["metadata"].items():
                    if isinstance(value, str) and word in value.lower():
                        score += 2
            
            # Special handling for count queries
            if "how many" in query_lower or "count" in query_lower:
                # Boost documents that match the counting criteria
                if "saudi" in query_lower and "saudi arabian" in text_lower:
                    score += 25
                elif "patient" in query_lower and doc["type"] == "patient":
                    score += 15
                elif "equipment" in query_lower and doc["type"] == "equipment":
                    score += 15
            
            if score > 0:
                doc_result = doc.copy()
                doc_result["relevance_score"] = score
                results.append(doc_result)
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:max_results]
    
    async def generate_ai_response(self, query: str, context_docs: List[Dict[str, Any]], model: str = "llama3.2:3b") -> Dict[str, Any]:
        """Generate AI response using Ollama"""
        
        # Build context from documents
        context = ""
        if context_docs:
            context = "\n--- HOSPITAL DATA ---\n"
            for i, doc in enumerate(context_docs[:8], 1):  # Limit context size
                context += f"\nDocument {i}:\n{doc['text']}\n"
        
        # Build RAG prompt
        prompt = f"""You are a helpful medical AI assistant for a hospital system. Use the provided hospital data to answer questions accurately and precisely.

{context}

Question: {query}

Instructions:
- Use ONLY the information provided in the hospital data above
- When counting, count carefully from the actual data provided
- Be specific and include relevant details
- If the data doesn't contain enough information, say so clearly
- For medical queries, include appropriate disclaimers
- Format your response clearly with proper structure

Answer:"""
        
        # Call Ollama API
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2000,
                        "top_p": 0.9
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "response": data.get("response", "").strip(),
                            "model": model,
                            "success": True
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error {response.status}: {error_text}")
                        return {
                            "response": self._generate_fallback_response(query, context_docs),
                            "model": "fallback",
                            "success": False,
                            "error": f"API Error: {response.status}"
                        }
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {
                "response": self._generate_fallback_response(query, context_docs),
                "model": "fallback", 
                "success": False,
                "error": str(e)
            }
    
    def _generate_fallback_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate fallback response when AI is unavailable"""
        query_lower = query.lower()
        
        # Saudi patients query
        if "how many" in query_lower and "saudi" in query_lower:
            saudi_docs = [doc for doc in context_docs if "saudi arabian" in doc["text"].lower()]
            count = len(saudi_docs)
            response = f"🏥 Based on the hospital database, we have **{count} Saudi patients**.\n\n"
            
            if saudi_docs:
                response += "Sample Saudi patients:\n"
                for i, doc in enumerate(saudi_docs[:5], 1):
                    name = doc["metadata"]["name"]
                    patient_id = doc["metadata"]["patient_id"]
                    status = doc["metadata"]["status"]
                    response += f"{i}. {name} (ID: {patient_id}, Status: {status})\n"
                
                if count > 5:
                    response += f"... and {count - 5} more Saudi patients.\n"
            
            response += f"\nSources: {len(context_docs)} hospital records"
            response += "\n\n⚠️ This response was generated using fallback logic as the AI model is unavailable."
            return response
        
        # Beds availability query
        elif "bed" in query_lower and ("available" in query_lower or "free" in query_lower):
            dept_docs = [doc for doc in context_docs if doc["type"] == "department"]
            total_available = sum(doc["metadata"].get("available_beds", 0) for doc in dept_docs)
            total_beds = sum(doc["metadata"].get("total_beds", 0) for doc in dept_docs)
            
            response = f"🛏️ **Hospital Bed Status:**\n"
            response += f"- Total Available Beds: **{total_available}**\n"
            response += f"- Total Beds: **{total_beds}**\n"
            response += f"- Occupancy Rate: **{((total_beds - total_available) / total_beds * 100):.1f}%**\n\n"
            
            if dept_docs:
                response += "Department Breakdown:\n"
                for dept in dept_docs[:5]:
                    name = dept["metadata"]["department_name"]
                    available = dept["metadata"].get("available_beds", 0)
                    total = dept["metadata"].get("total_beds", 0)
                    response += f"- {name}: {available}/{total} beds available\n"
            
            response += f"\nSources: {len(context_docs)} hospital records"
            response += "\n\n⚠️ This response was generated using fallback logic as the AI model is unavailable."
            return response
        
        # Doctor queries (any doctor-related question)
        elif "doctor" in query_lower or "specialist" in query_lower or "physician" in query_lower:
            doctor_docs = [doc for doc in context_docs if doc["type"] == "doctor"]
            
            # Extract specialty keywords from query
            specialty_keywords = {
                "orthopedic": ["bone", "orthopedic", "ortho"],
                "cardiology": ["heart", "cardio", "cardiac"],
                "neurosurgery": ["brain", "neuro", "surgeon"],
                "pediatrics": ["child", "pediatric", "pediatrics"],
                "emergency": ["emergency", "er"],
                "radiology": ["radiology", "radio", "imaging"],
                "internal": ["internal", "medicine"],
                "anesthesiology": ["anesthesia", "anesthesiology"],
                "gastroenterology": ["gastro", "stomach", "digestive"]
            }
            
            relevant_doctors = []
            
            # Find doctors by specialty matching
            for specialty, keywords in specialty_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    for doc in doctor_docs:
                        doc_specialty = doc["metadata"]["specialty"].lower()
                        if specialty in doc_specialty or any(keyword in doc_specialty for keyword in keywords):
                            if doc not in relevant_doctors:
                                relevant_doctors.append(doc)
            
            # If no specific specialty match, return all doctors from search results
            if not relevant_doctors:
                relevant_doctors = doctor_docs
            
            if relevant_doctors:
                response = f"🩺 Found **{len(relevant_doctors)} doctors** matching your query:\n\n"
                for i, doc in enumerate(relevant_doctors[:5], 1):
                    name = doc["metadata"]["name"]
                    specialty = doc["metadata"]["specialty"]
                    department = doc["metadata"]["department"]
                    experience = doc["metadata"]["experience"]
                    availability = doc["metadata"]["availability"]
                    response += f"{i}. {name}\n"
                    response += f"   - Specialty: {specialty}\n"
                    response += f"   - Department: {department}\n"
                    response += f"   - Experience: {experience} years\n"
                    response += f"   - Status: {availability}\n\n"
                
                if len(relevant_doctors) > 5:
                    response += f"... and {len(relevant_doctors) - 5} more doctors.\n"
            else:
                response = f"🔍 No doctors found in search results.\n\n"
                response += f"Available specialties include: Cardiology, Emergency Medicine, Neurosurgery, Pediatrics, Orthopedics, Radiology, Internal Medicine, Anesthesiology, Gastroenterology, etc."
            
            response += f"\nSources: {len(context_docs)} hospital records"
            response += "\n\n⚠️ This response was generated using fallback logic as the AI model is unavailable."
            return response
        
        # Department capacity queries
        elif "department" in query_lower and ("capacity" in query_lower or "emergency" in query_lower):
            dept_docs = [doc for doc in context_docs if doc["type"] == "department"]
            
            # Filter for specific department if mentioned
            if "emergency" in query_lower:
                dept_docs = [doc for doc in dept_docs if "emergency" in doc["metadata"]["department_name"].lower()]
            
            response = f"🏥 **Department Information:**\n\n"
            
            for dept in dept_docs[:3]:  # Show top 3 departments
                name = dept["metadata"]["department_name"]
                total_beds = dept["metadata"]["total_beds"]
                available_beds = dept["metadata"]["available_beds"]
                occupied_beds = dept["metadata"]["occupied_beds"]
                head = dept["metadata"]["head_of_department"]
                location = dept["metadata"]["location"]
                floor = dept["metadata"]["floor"]
                
                response += f"**{name}:**\n"
                response += f"- Total Beds: {total_beds}\n"
                response += f"- Available Beds: {available_beds}\n"
                response += f"- Occupied Beds: {occupied_beds}\n"
                response += f"- Occupancy Rate: {(occupied_beds/total_beds*100):.1f}%\n"
                response += f"- Head of Department: {head}\n"
                response += f"- Location: {location}, Floor {floor}\n\n"
            
            response += f"Sources: {len(context_docs)} hospital records"
            response += "\n⚠️ This response was generated using fallback logic as the AI model is unavailable."
            return response
        
        # General patient count
        elif "how many" in query_lower and "patient" in query_lower:
            patient_docs = [doc for doc in context_docs if doc["type"] == "patient"]
            count = len(patient_docs)
            return f"🏥 Based on the hospital database, we have **{count} total patients**.\n\nSources: {len(context_docs)} hospital records\n\n⚠️ This response was generated using fallback logic as the AI model is unavailable."
        
        # General fallback
        else:
            doc_types = {}
            for doc in context_docs:
                doc_type = doc["type"]
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            response = f"📊 Found {len(context_docs)} relevant documents for: '{query}'\n\n"
            if doc_types:
                response += "Document types found:\n"
                for doc_type, count in doc_types.items():
                    response += f"- {doc_type}: {count} records\n"
            
            response += "\n⚠️ AI model is unavailable. Please ensure Ollama is running with a compatible model."
            return response
    
    def initialize(self):
        """Initialize the RAG system"""
        if not self.initialized:
            logger.info("Initializing Simple RAG Engine...")
            self.documents = self.load_hospital_data()
            self.initialized = True
            logger.info(f"RAG engine initialized with {len(self.documents)} documents")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.initialized:
            return {"error": "Not initialized"}
        
        type_counts = {}
        for doc in self.documents:
            doc_type = doc["type"]
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "type_distribution": type_counts,
            "system": "Simple RAG Engine",
            "status": "ready"
        }

# Global instance
simple_rag = None

def get_rag_engine(data_dir: Path) -> SimpleRAGEngine:
    """Get or create RAG engine instance"""
    global simple_rag
    if simple_rag is None:
        simple_rag = SimpleRAGEngine(data_dir)
        simple_rag.initialize()
    return simple_rag

if __name__ == "__main__":
    # Test the simple RAG engine
    import asyncio
    
    async def test_rag():
        data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
        rag = SimpleRAGEngine(data_dir)
        rag.initialize()
        
        print("🔍 Testing Simple RAG Engine")
        print("=" * 40)
        
        # Test search
        query = "How many Saudi patients do we have?"
        print(f"Query: {query}")
        
        results = rag.semantic_search(query, max_results=100)
        print(f"Found {len(results)} documents")
        
        # Test AI response
        response = await rag.generate_ai_response(query, results)
        print("\nAI Response:")
        print("-" * 20)
        print(response["response"])
        
        # Stats
        stats = rag.get_stats()
        print(f"\nSystem Stats: {stats}")
    
    asyncio.run(test_rag())