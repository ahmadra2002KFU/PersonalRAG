"""
Real RAG Engine with Embeddings and Vector Search
Proper implementation using ChromaDB and Sentence Transformers
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class HospitalRAGEngine:
    """Real RAG engine with semantic search capabilities"""
    
    def __init__(self, data_dir: Path, persist_directory: str = "./chroma_db"):
        self.data_dir = Path(data_dir)
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="hospital_data",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized successfully")
        
    def load_and_process_csv_data(self) -> List[Dict[str, Any]]:
        """Load CSV files and create documents for embedding"""
        documents = []
        
        # Process patient data
        try:
            patients_df = pd.read_csv(self.data_dir / "hospital_patients.csv")
            logger.info(f"Loaded {len(patients_df)} patient records")
            
            for _, patient in patients_df.iterrows():
                # Create comprehensive document text
                doc_text = f"""
                Patient ID: {patient['Patient_ID']}
                Name: {patient['First_Name']} {patient['Last_Name']}
                Date of Birth: {patient['Date_of_Birth']}
                Gender: {patient['Gender']}
                Nationality: {patient.get('Nationality', 'Unknown')}
                Phone: {patient.get('Phone', 'Unknown')}
                Email: {patient.get('Email', 'Unknown')}
                Address: {patient.get('Address', 'Unknown')}
                Insurance Type: {patient.get('Insurance_Type', 'Unknown')}
                Blood Type: {patient.get('Blood_Type', 'Unknown')}
                Allergies: {patient.get('Allergies', 'None')}
                Current Status: {patient.get('Current_Status', 'Unknown')}
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
                        "blood_type": patient.get('Blood_Type', 'Unknown'),
                        "insurance": patient.get('Insurance_Type', 'Unknown')
                    }
                })
                
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
        
        # Process equipment data
        try:
            equipment_df = pd.read_csv(self.data_dir / "hospital_equipment.csv")
            logger.info(f"Loaded {len(equipment_df)} equipment records")
            
            for _, equipment in equipment_df.iterrows():
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
                        "manufacturer": equipment['Manufacturer']
                    }
                })
                
        except Exception as e:
            logger.error(f"Error loading equipment data: {e}")
        
        # Process medical records
        try:
            records_df = pd.read_csv(self.data_dir / "hospital_medical_records.csv")
            logger.info(f"Loaded {len(records_df)} medical records")
            
            for _, record in records_df.iterrows():
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
                        "date": record['Date']
                    }
                })
                
        except Exception as e:
            logger.error(f"Error loading medical records: {e}")
        
        # Process appointments
        try:
            appointments_df = pd.read_csv(self.data_dir / "hospital_appointments.csv")
            logger.info(f"Loaded {len(appointments_df)} appointment records")
            
            for _, appointment in appointments_df.iterrows():
                doc_text = f"""
                Appointment ID: {appointment['Appointment_ID']}
                Patient ID: {appointment['Patient_ID']}
                Doctor ID: {appointment['Doctor_ID']}
                Department: {appointment['Department']}
                Date: {appointment['Date']}
                Time: {appointment['Time']}
                Appointment Type: {appointment['Appointment_Type']}
                Status: {appointment['Status']}
                Duration: {appointment.get('Duration_Minutes', 'Unknown')} minutes
                Reason: {appointment.get('Reason', 'N/A')}
                """.strip()
                
                documents.append({
                    "id": f"appointment_{appointment['Appointment_ID']}",
                    "text": doc_text,
                    "type": "appointment",
                    "metadata": {
                        "appointment_id": appointment['Appointment_ID'],
                        "patient_id": appointment['Patient_ID'],
                        "doctor_id": appointment['Doctor_ID'],
                        "department": appointment['Department'],
                        "status": appointment['Status'],
                        "date": appointment['Date']
                    }
                })
                
        except Exception as e:
            logger.error(f"Error loading appointment data: {e}")
        
        # Process operational metrics
        try:
            metrics_df = pd.read_csv(self.data_dir / "hospital_operational_metrics.csv")
            logger.info(f"Loaded {len(metrics_df)} operational metrics")
            
            for _, metric in metrics_df.iterrows():
                doc_text = f"""
                Metric Category: {metric['Metric_Category']}
                Metric Name: {metric['Metric_Name']}
                Department: {metric['Department']}
                Current Value: {metric['Current_Value']} {metric['Unit']}
                Target Value: {metric['Target_Value']} {metric['Unit']}
                Trend Direction: {metric.get('Trend_Direction', 'Stable')}
                Last Updated: {metric.get('Last_Updated', 'Unknown')}
                Notes: {metric.get('Notes', 'N/A')}
                """.strip()
                
                documents.append({
                    "id": f"metric_{metric['Metric_Category']}_{metric['Metric_Name']}".replace(' ', '_').lower(),
                    "text": doc_text,
                    "type": "metric",
                    "metadata": {
                        "category": metric['Metric_Category'],
                        "metric_name": metric['Metric_Name'],
                        "department": metric['Department'],
                        "current_value": metric['Current_Value'],
                        "target_value": metric['Target_Value']
                    }
                })
                
        except Exception as e:
            logger.error(f"Error loading metrics data: {e}")
        
        logger.info(f"Total documents created: {len(documents)}")
        return documents
    
    def create_embeddings_and_store(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Create embeddings and store in ChromaDB"""
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        
        # Clear existing collection
        try:
            self.chroma_client.delete_collection("hospital_data")
            self.collection = self.chroma_client.create_collection(
                name="hospital_data",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared existing collection")
        except:
            pass
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract texts and create embeddings
            texts = [doc["text"] for doc in batch]
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Prepare data for ChromaDB
            ids = [doc["id"] for doc in batch]
            metadatas = []
            
            for doc in batch:
                metadata = doc["metadata"].copy()
                metadata["type"] = doc["type"]
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        logger.info("All embeddings created and stored successfully!")
    
    def semantic_search(self, query: str, n_results: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build where clause for filtering
        where_clause = None
        if filter_dict:
            where_clause = filter_dict
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def initialize_database(self):
        """Initialize the RAG database with hospital data"""
        logger.info("Initializing RAG database...")
        
        # Check if already initialized
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"Database already contains {count} documents")
                return
        except:
            pass
        
        # Load and process data
        documents = self.load_and_process_csv_data()
        
        if not documents:
            logger.error("No documents loaded!")
            return
        
        # Create embeddings and store
        self.create_embeddings_and_store(documents)
        
        logger.info("RAG database initialization complete!")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            count = self.collection.count()
            
            # Get type distribution
            results = self.collection.get(include=["metadatas"])
            type_counts = {}
            
            for metadata in results["metadatas"]:
                doc_type = metadata.get("type", "unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            return {
                "total_documents": count,
                "type_distribution": type_counts,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_database": "ChromaDB"
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test the RAG engine
    import sys
    
    data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
    rag_engine = HospitalRAGEngine(data_dir)
    
    # Initialize database
    rag_engine.initialize_database()
    
    # Test search
    results = rag_engine.semantic_search("How many Saudi patients do we have?", n_results=5)
    
    print("\nSearch Results:")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Similarity: {result['similarity_score']:.3f}")
        print(f"Content: {result['document'][:200]}...")
        print("-" * 50)
    
    # Print stats
    stats = rag_engine.get_database_stats()
    print("\nDatabase Stats:")
    print(json.dumps(stats, indent=2))