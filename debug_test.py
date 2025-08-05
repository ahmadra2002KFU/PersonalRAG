"""
Debug specific test case
"""

import asyncio
from pathlib import Path
from hospital_rag.backend.simple_rag import SimpleRAGEngine

async def debug_emergency_test():
    print("🔍 Debugging Emergency Department Test")
    print("=" * 40)
    
    data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
    rag = SimpleRAGEngine(data_dir)
    rag.initialize()
    
    query = "Tell me about the Emergency Department capacity"
    results = rag.semantic_search(query, max_results=30)
    
    print(f"🔍 Query: {query}")
    print(f"📊 Found {len(results)} documents")
    
    # Show document types
    doc_types = {}
    for doc in results:
        doc_type = doc["type"]
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    print(f"📋 Document types: {doc_types}")
    
    # Generate response
    response = await rag.generate_ai_response(query, results)
    
    print(f"\n📝 Full Response:")
    print("-" * 30)
    print(response['response'])
    
    # Check if "25 beds" is in response
    has_25_beds = "25 beds" in response['response'].lower()
    has_total_25 = "total beds: 25" in response['response'].lower()
    
    print(f"\n🔍 Analysis:")
    print(f"✅ Contains '25 beds': {has_25_beds}")
    print(f"✅ Contains 'Total Beds: 25': {has_total_25}")
    print(f"✅ Response length: {len(response['response'])} characters")

if __name__ == "__main__":
    asyncio.run(debug_emergency_test())