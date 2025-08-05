"""
Verify Saudi Patient Count in RAG System
Quick test to check if all Saudi patients are being found
"""

from hospital_rag.backend.simple_rag import SimpleRAGEngine
from pathlib import Path

def test_saudi_count():
    print("🏥 Testing Saudi Patient Count in RAG System")
    print("=" * 50)
    
    # Initialize RAG engine
    data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
    rag = SimpleRAGEngine(data_dir)
    rag.initialize()
    
    print(f"📊 Total documents loaded: {len(rag.documents)}")
    
    # Count patient documents
    patient_docs = [doc for doc in rag.documents if doc["type"] == "patient"]
    print(f"👥 Total patient documents: {len(patient_docs)}")
    
    # Count Saudi patients in loaded documents
    saudi_docs = [doc for doc in patient_docs if "saudi arabian" in doc["text"].lower()]
    print(f"🇸🇦 Saudi patients in documents: {len(saudi_docs)}")
    
    # Test search with higher limit
    query = "How many Saudi patients do we have?"
    search_results = rag.semantic_search(query, max_results=100)  # Increase limit
    
    print(f"\n🔍 Search Results:")
    print(f"📋 Total search results: {len(search_results)}")
    
    # Filter search results for Saudi patients
    saudi_search_results = [doc for doc in search_results if "saudi arabian" in doc["text"].lower()]
    print(f"🇸🇦 Saudi patients in search results: {len(saudi_search_results)}")
    
    # Show document types in search results
    doc_types = {}
    for doc in search_results:
        doc_type = doc["type"]
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print(f"📊 Search result types: {doc_types}")
    
    # Test the fallback response
    import asyncio
    
    async def test_response():
        response = await rag.generate_ai_response(query, search_results)
        print(f"\n🤖 AI Response:")
        print("-" * 30)
        print(response["response"])
    
    asyncio.run(test_response())
    
    # Verify against raw CSV
    import csv
    with open(data_dir / "hospital_patients.csv", 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        csv_saudi_count = sum(1 for row in reader if "saudi" in row.get('Nationality', '').lower())
    
    print(f"\n✅ Verification:")
    print(f"   Raw CSV: {csv_saudi_count} Saudi patients")
    print(f"   RAG docs: {len(saudi_docs)} Saudi patients") 
    print(f"   Search: {len(saudi_search_results)} Saudi patients found")
    print(f"   Status: {'CORRECT' if len(saudi_docs) == csv_saudi_count else 'MISMATCH'}")

if __name__ == "__main__":
    test_saudi_count()