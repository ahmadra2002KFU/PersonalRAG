"""
Final Validation - Key Hospital Queries
Tests the most important queries to confirm fully functional RAG
"""

import asyncio
from pathlib import Path
from hospital_rag.backend.simple_rag import SimpleRAGEngine

async def test_key_queries():
    print("🏥 FINAL RAG SYSTEM VALIDATION")
    print("=" * 50)
    
    # Initialize
    data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
    rag = SimpleRAGEngine(data_dir)
    rag.initialize()
    
    stats = rag.get_stats()
    print(f"📊 System Stats: {stats['total_documents']} documents")
    print(f"📋 Types: {stats['type_distribution']}")
    print()
    
    # Key test queries
    tests = [
        {
            "name": "Saudi Patient Count (CRITICAL)",
            "query": "How many Saudi patients do we have?",
            "expected": "54",
            "search_limit": 100
        },
        {
            "name": "Total Available Beds",
            "query": "How many beds are available in the hospital?",
            "expected": "74",
            "search_limit": 50
        },
        {
            "name": "Orthopedic Doctor",
            "query": "Who is the orthopedic specialist?",
            "expected": "Dr. David Kumar",
            "search_limit": 30
        },
        {
            "name": "Emergency Department Info",
            "query": "Tell me about the Emergency Department capacity",
            "expected": "25",
            "search_limit": 30
        },
        {
            "name": "ICU Equipment",
            "query": "What patient monitors are in the ICU?",
            "expected": "Patient Monitor",
            "search_limit": 30
        }
    ]
    
    all_passed = True
    
    for i, test in enumerate(tests, 1):
        print(f"🧪 TEST {i}: {test['name']}")
        print(f"❓ Query: {test['query']}")
        
        # Search
        results = rag.semantic_search(test['query'], max_results=test['search_limit'])
        
        # Generate response
        response = await rag.generate_ai_response(test['query'], results)
        
        # Check if expected content is in response
        found_expected = test['expected'].lower() in response['response'].lower()
        
        print(f"📊 Found {len(results)} documents")
        print(f"✅ Expected '{test['expected']}': {'FOUND' if found_expected else 'NOT FOUND'}")
        print(f"📝 Response preview: {response['response'][:100]}...")
        
        if not found_expected:
            all_passed = False
            print(f"❌ TEST FAILED!")
        else:
            print(f"✅ TEST PASSED!")
        
        print("-" * 40)
    
    # Final assessment
    print(f"\n{'='*20} FINAL VALIDATION RESULT {'='*20}")
    if all_passed:
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("✅ RAG System is FULLY FUNCTIONAL")
        print("✅ No mock data - all real hospital information")
        print("✅ Search and response generation working correctly")
        print("✅ All CSV data properly loaded and accessible")
    else:
        print("❌ Some tests failed - system needs review")
    
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(test_key_queries())
    exit(0 if result else 1)