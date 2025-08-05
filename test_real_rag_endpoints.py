"""
Test the real RAG endpoints to show the correct results
"""

import requests
import json

def test_rag_endpoints():
    print("🏥 TESTING REAL RAG SYSTEM ENDPOINTS")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test 1: System info
    print("🔍 TEST 1: System Info")
    try:
        response = requests.get(f"{base_url}/")
        data = response.json()
        print(f"✅ System: {data['system_type']}")
        print(f"📊 Documents: {data['total_documents']}")
        print(f"🧮 Embeddings: {data['embedding_dimensions']}D")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 2: Saudi patients (vector search)
    print("🔍 TEST 2: Saudi Patients (Vector Search)")
    try:
        response = requests.get(f"{base_url}/api/test/saudi-patients")
        data = response.json()
        print(f"✅ Method: {data['method']}")
        print(f"📊 Documents searched: {data['total_documents_searched']}")
        print(f"🇸🇦 Saudi patients found: {data['saudi_patients_found']}")
        
        if data['sample_patients']:
            print("📋 Sample results:")
            for i, patient in enumerate(data['sample_patients'][:3], 1):
                # Extract name from text
                lines = patient['text'].split('\\n')
                name_line = lines[0] if lines else patient['text'][:50]
                print(f"   {i}. {name_line}")
                print(f"      Similarity: {patient['similarity']:.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 3: Vector search directly
    print("🔍 TEST 3: Vector Search for Saudi Patients")
    try:
        payload = {
            "query": "Saudi Arabian nationality patients",
            "top_k": 30
        }
        response = requests.post(f"{base_url}/api/search", json=payload)
        data = response.json()
        
        # Filter for patient documents
        patient_docs = [doc for doc in data['results'] if doc['type'] == 'patient']
        saudi_patients = [doc for doc in patient_docs if 'saudi arabian' in doc['text'].lower()]
        
        print(f"✅ Total results: {len(data['results'])}")
        print(f"👥 Patient documents: {len(patient_docs)}")
        print(f"🇸🇦 Saudi patients: {len(saudi_patients)}")
        print(f"📊 Search method: {data['search_method']}")
        
        if saudi_patients:
            print("📋 Top Saudi patients found:")
            for i, patient in enumerate(saudi_patients[:3], 1):
                lines = patient['text'].split('\\n')
                name_line = lines[0] if lines else patient['text'][:50]
                print(f"   {i}. {name_line}")
                print(f"      Similarity: {patient['similarity']:.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Test 4: Compare with orthopedic doctors
    print("🔍 TEST 4: Orthopedic Doctors")
    try:
        response = requests.get(f"{base_url}/api/test/orthopedic-doctors")
        data = response.json()
        print(f"✅ Doctors found: {data['doctors_found']}")
        
        if data['doctors']:
            print("📋 Sample doctors:")
            for i, doctor in enumerate(data['doctors'][:2], 1):
                lines = doctor['text'].split('\\n')
                name_line = lines[0] if lines else doctor['text'][:50]
                specialty_line = next((line for line in lines if 'Specialty:' in line), '')
                print(f"   {i}. {name_line}")
                if specialty_line:
                    print(f"      {specialty_line}")
                print(f"      Similarity: {doctor['similarity']:.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    print("=" * 60)
    print("📋 INSTRUCTIONS TO USE REAL RAG SYSTEM:")
    print("=" * 60)
    print("🌐 Make sure you're using: http://localhost:8000")
    print("🔍 Saudi patients: GET http://localhost:8000/api/test/saudi-patients")
    print("👨‍⚕️ Doctors: GET http://localhost:8000/api/test/orthopedic-doctors")
    print("🔎 Vector search: POST http://localhost:8000/api/search")
    print("🤖 Full RAG: POST http://localhost:8000/api/rag")
    print("📊 System info: GET http://localhost:8000/")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_endpoints()