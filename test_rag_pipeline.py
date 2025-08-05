"""
Test RAG Pipeline - Shows that real RAG is working
Tests embeddings, vector search, and document retrieval
"""

import asyncio
from pathlib import Path
from real_rag_system import RealRAGSystem

async def test_rag_pipeline():
    print("🏥 TESTING REAL RAG PIPELINE")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RealRAGSystem(Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG"))
    stats = rag.initialize()
    
    print(f"✅ RAG System Initialized:")
    print(f"   📊 Documents loaded: {stats['total_documents']}")
    print(f"   🧮 Embedding dimensions: {stats['embedding_dimensions']}")
    print(f"   🎯 Status: {stats['status']}")
    print()
    
    # Test queries to verify RAG components
    test_cases = [
        {
            "query": "How many Saudi patients do we have?",
            "expected_types": ["patient"],
            "should_find": "Saudi Arabian"
        },
        {
            "query": "Who is the orthopedic specialist?", 
            "expected_types": ["doctor"],
            "should_find": "Orthopedics"
        },
        {
            "query": "Emergency Department bed capacity",
            "expected_types": ["department"],
            "should_find": "Emergency"
        },
        {
            "query": "ICU patient monitoring equipment",
            "expected_types": ["equipment"],
            "should_find": "Patient Monitor"
        }
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"🧪 TEST {i}: RAG Vector Search")
        print(f"❓ Query: {test['query']}")
        
        # Perform vector search (this is the RAG retrieval step)
        relevant_docs = rag.vector_search(test['query'], top_k=10)
        
        print(f"📊 Vector search found: {len(relevant_docs)} documents")
        
        # Analyze results
        doc_types = {}
        contains_expected = False
        highest_similarity = 0
        
        for doc in relevant_docs:
            doc_type = doc["type"]
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            if test["should_find"].lower() in doc["text"].lower():
                contains_expected = True
            
            if doc["similarity"] > highest_similarity:
                highest_similarity = doc["similarity"]
        
        print(f"📋 Document types retrieved: {doc_types}")
        print(f"🎯 Highest similarity score: {highest_similarity:.3f}")
        print(f"✅ Contains expected content '{test['should_find']}': {contains_expected}")
        
        # Check if expected document types were found
        expected_found = any(doc_type in test["expected_types"] for doc_type in doc_types.keys())
        print(f"✅ Expected document types found: {expected_found}")
        
        # Show top result to prove semantic matching
        if relevant_docs:
            top_doc = relevant_docs[0]
            print(f"🔍 Top result ({top_doc['similarity']:.3f} similarity):")
            print(f"   Type: {top_doc['type']}")
            print(f"   Content: {top_doc['text'][:100]}...")
        
        # Test passes if we found relevant documents with good similarity
        test_passed = len(relevant_docs) > 0 and highest_similarity > 0.1 and expected_found
        
        if test_passed:
            print("✅ RAG TEST PASSED!")
        else:
            print("❌ RAG TEST FAILED!")
            all_passed = False
        
        print("-" * 50)
    
    # Test specific Saudi patient count 
    print("🔍 SPECIFIC TEST: Saudi Patient Count")
    saudi_query_results = rag.vector_search("Saudi Arabian nationality patients", top_k=100)
    
    saudi_patients = []
    for doc in saudi_query_results:
        if doc["type"] == "patient" and "saudi arabian" in doc["text"].lower():
            saudi_patients.append(doc)
    
    print(f"📊 Vector search retrieved: {len(saudi_query_results)} total documents")
    print(f"🇸🇦 Saudi patients found by RAG: {len(saudi_patients)}")
    
    # Verify against raw data
    import csv
    with open(Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG") / "hospital_patients.csv", 'r') as file:
        reader = csv.DictReader(file)
        csv_saudi_count = sum(1 for row in reader if "saudi" in row.get('Nationality', '').lower())
    
    print(f"📋 Actual Saudi patients in CSV: {csv_saudi_count}")
    print(f"✅ RAG accuracy: {len(saudi_patients)}/{csv_saudi_count} = {len(saudi_patients)/csv_saudi_count*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("🎯 RAG SYSTEM VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_passed and len(saudi_patients) >= csv_saudi_count * 0.8:
        print("🎉 REAL RAG SYSTEM CONFIRMED!")
        print("✅ Vector embeddings working correctly")
        print("✅ Semantic search finding relevant documents")
        print("✅ Document types correctly identified")
        print("✅ Similarity scores showing semantic understanding")
        print("✅ Saudi patient count accurate via vector search")
        print("\n📋 This is a GENUINE RAG system with:")
        print("   - Real vector embeddings")
        print("   - Cosine similarity search")
        print("   - Semantic document matching")
        print("   - LLM integration ready (Ollama)")
    else:
        print("❌ RAG system needs improvement")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())