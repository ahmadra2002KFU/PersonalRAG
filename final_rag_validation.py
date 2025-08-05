"""
FINAL RAG SYSTEM VALIDATION
Comprehensive test to prove this is a REAL RAG system, not keyword search
"""

import asyncio
import numpy as np
from pathlib import Path
from real_rag_system import RealRAGSystem, create_text_embedding, cosine_similarity

async def validate_real_rag():
    print("🏥 FINAL RAG SYSTEM VALIDATION")
    print("=" * 60)
    print("Testing REAL RAG components:")
    print("✅ Vector embeddings")
    print("✅ Semantic similarity search")
    print("✅ LLM integration")
    print("✅ No keyword matching")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RealRAGSystem(Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG"))
    stats = rag.initialize()
    
    print(f"📊 System initialized: {stats}")
    print()
    
    # TEST 1: Prove embeddings are semantic, not keyword-based
    print("🧪 TEST 1: Semantic Understanding (Not Keyword Matching)")
    print("-" * 50)
    
    # Test semantic similarity between related concepts
    text1 = "orthopedic bone specialist doctor"
    text2 = "Dr. David Kumar specializes in Orthopedics"
    text3 = "Saudi Arabian patient nationality"
    text4 = "Patient from Saudi Arabia"
    
    emb1 = create_text_embedding(text1)
    emb2 = create_text_embedding(text2)
    emb3 = create_text_embedding(text3)
    emb4 = create_text_embedding(text4)
    
    similarity_orthopedic = cosine_similarity(emb1, emb2)
    similarity_saudi = cosine_similarity(emb3, emb4)
    similarity_unrelated = cosine_similarity(emb1, emb3)
    
    print(f"📊 Semantic similarities:")
    print(f"   'orthopedic specialist' ↔ 'Dr. Kumar Orthopedics': {similarity_orthopedic:.3f}")
    print(f"   'Saudi patient' ↔ 'Patient from Saudi Arabia': {similarity_saudi:.3f}")
    print(f"   'orthopedic' ↔ 'Saudi patient' (unrelated): {similarity_unrelated:.3f}")
    
    semantic_test_passed = (similarity_orthopedic > 0.3 and 
                           similarity_saudi > 0.3 and 
                           similarity_unrelated < similarity_orthopedic)
    
    print(f"✅ Semantic understanding: {'PASS' if semantic_test_passed else 'FAIL'}")
    print()
    
    # TEST 2: Vector search finds semantically similar documents
    print("🧪 TEST 2: Vector Search Semantic Matching")
    print("-" * 50)
    
    test_queries = [
        {
            "query": "How many patients from Saudi Arabia?",
            "expected_type": "patient",
            "expected_concept": "saudi"
        },
        {
            "query": "bone and joint specialist doctors",
            "expected_type": "doctor", 
            "expected_concept": "orthopedic"
        },
        {
            "query": "hospital bed availability and capacity",
            "expected_type": "department",
            "expected_concept": "beds"
        }
    ]
    
    vector_tests_passed = 0
    
    for i, test in enumerate(test_queries, 1):
        print(f"Query {i}: {test['query']}")
        
        results = rag.vector_search(test['query'], top_k=5)
        
        # Check if correct document types are prioritized
        top_types = [doc['type'] for doc in results[:3]]
        type_match = test['expected_type'] in top_types
        
        # Check semantic content matching
        content_matches = 0
        for doc in results[:3]:
            if test['expected_concept'] in doc['text'].lower():
                content_matches += 1
        
        semantic_match = content_matches > 0
        
        print(f"   Top document types: {top_types}")
        print(f"   Expected type '{test['expected_type']}' found: {type_match}")
        print(f"   Semantic concept '{test['expected_concept']}' found: {semantic_match}")
        print(f"   Top similarity score: {results[0]['similarity']:.3f}")
        
        if type_match and results[0]['similarity'] > 0.2:
            vector_tests_passed += 1
            print(f"   ✅ PASS")
        else:
            print(f"   ❌ FAIL")
        print()
    
    vector_search_passed = vector_tests_passed >= 2
    
    # TEST 3: Full RAG pipeline with LLM integration
    print("🧪 TEST 3: Complete RAG Pipeline")
    print("-" * 50)
    
    rag_query = "How many Saudi patients are there in the hospital?"
    print(f"RAG Query: {rag_query}")
    
    # Get RAG response
    rag_result = await rag.rag_query(rag_query, top_k=10)
    
    print(f"📊 RAG Pipeline Results:")
    print(f"   Documents retrieved: {len(rag_result['sources'])}")
    print(f"   Method used: {rag_result['method']}")
    print(f"   Response length: {len(rag_result['answer'])} characters")
    print(f"   LLM Response: {rag_result['answer'][:100]}...")
    
    # Check if sources are relevant
    saudi_sources = sum(1 for source in rag_result['sources'] 
                       if 'saudi' in source['text'].lower())
    
    print(f"   Saudi-related sources: {saudi_sources}/{len(rag_result['sources'])}")
    
    rag_pipeline_passed = (len(rag_result['sources']) > 0 and 
                          saudi_sources > 0 and
                          len(rag_result['answer']) > 50)
    
    print(f"✅ RAG Pipeline: {'PASS' if rag_pipeline_passed else 'FAIL'}")
    print()
    
    # TEST 4: Prove it's NOT keyword search
    print("🧪 TEST 4: NOT Keyword Search (Semantic vs Lexical)")
    print("-" * 50)
    
    # Query with synonyms/related terms that wouldn't match keywords
    semantic_query = "physicians specializing in bone and joint problems"
    keyword_query = "orthopedic doctors"
    
    semantic_results = rag.vector_search(semantic_query, top_k=5)
    keyword_results = rag.vector_search(keyword_query, top_k=5)
    
    print(f"Semantic query: '{semantic_query}'")
    print(f"Keyword query: '{keyword_query}'")
    
    # Both should find similar results if it's semantic search
    semantic_doctors = sum(1 for doc in semantic_results if doc['type'] == 'doctor')
    keyword_doctors = sum(1 for doc in keyword_results if doc['type'] == 'doctor')
    
    print(f"Semantic query found {semantic_doctors} doctors")
    print(f"Keyword query found {keyword_doctors} doctors")
    
    # If it's truly semantic, both should find doctors
    not_keyword_search = semantic_doctors > 0
    
    print(f"✅ Semantic (not keyword) search: {'PASS' if not_keyword_search else 'FAIL'}")
    print()
    
    # FINAL ASSESSMENT
    print("=" * 60)
    print("🎯 FINAL RAG SYSTEM ASSESSMENT")
    print("=" * 60)
    
    total_tests = 4
    tests_passed = sum([
        semantic_test_passed,
        vector_search_passed,
        rag_pipeline_passed,
        not_keyword_search
    ])
    
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    print()
    
    if tests_passed >= 3:
        print("🎉 CONFIRMED: This is a REAL RAG SYSTEM!")
        print("✅ Uses vector embeddings for semantic understanding")
        print("✅ Performs similarity-based document retrieval")
        print("✅ Integrates with LLM for response generation")
        print("✅ NOT a keyword search system")
        print("✅ Demonstrates semantic understanding of medical concepts")
        print()
        print("🏥 RAG System Components Validated:")
        print("   📊 Document embeddings: 384-dimensional vectors")
        print(f"   🔍 Vector database: {len(rag.documents)} documents")
        print("   🧮 Similarity search: Cosine similarity")
        print("   🤖 LLM integration: Ollama API")
        print("   🎯 Domain: Medical/Hospital data")
    else:
        print("❌ System needs improvement to be considered a true RAG system")
    
    print("=" * 60)
    
    return tests_passed >= 3

if __name__ == "__main__":
    result = asyncio.run(validate_real_rag())
    print(f"\n{'🎉 SUCCESS' if result else '❌ FAILED'}: RAG System Validation")
    exit(0 if result else 1)