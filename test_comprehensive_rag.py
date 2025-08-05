"""
Comprehensive RAG System Test
Tests 10+ real hospital queries to ensure NO MOCK responses
"""

import asyncio
import json
import time
from pathlib import Path
from hospital_rag.backend.simple_rag import SimpleRAGEngine

class RAGTester:
    def __init__(self):
        self.data_dir = Path("/mnt/z/Code/AI SIMA-RAG/0.3Ver/PersonalRAG")
        self.rag_engine = SimpleRAGEngine(self.data_dir)
        self.test_results = []
        
    def initialize(self):
        """Initialize the RAG system"""
        print("🏥 Initializing Hospital RAG System...")
        print("=" * 60)
        self.rag_engine.initialize()
        
        stats = self.rag_engine.get_stats()
        print(f"✅ System initialized successfully!")
        print(f"📊 Total documents loaded: {stats['total_documents']}")
        print(f"📋 Document types: {stats['type_distribution']}")
        print("=" * 60)
    
    async def run_test(self, test_name: str, query: str, expected_keywords: list = None):
        """Run a single test query"""
        print(f"\n🧪 TEST: {test_name}")
        print(f"🔍 Query: {query}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Perform search
        search_results = self.rag_engine.semantic_search(query, max_results=20)
        search_time = time.time() - start_time
        
        # Generate response
        ai_response = await self.rag_engine.generate_ai_response(query, search_results)
        total_time = time.time() - start_time
        
        # Analyze results
        result = {
            "test_name": test_name,
            "query": query,
            "search_results_count": len(search_results),
            "search_time": round(search_time, 3),
            "total_time": round(total_time, 3),
            "ai_success": ai_response.get("success", False),
            "model_used": ai_response.get("model", "fallback"),
            "response": ai_response.get("response", ""),
            "has_real_data": False,
            "keywords_found": [],
            "document_types": {}
        }
        
        # Count document types in results
        for doc in search_results:
            doc_type = doc["type"]
            result["document_types"][doc_type] = result["document_types"].get(doc_type, 0) + 1
        
        # Check for expected keywords in response
        response_lower = result["response"].lower()
        if expected_keywords:
            for keyword in expected_keywords:
                if keyword.lower() in response_lower:
                    result["keywords_found"].append(keyword)
        
        # Check if response contains real data (not mock)
        real_data_indicators = ["based on the hospital database", "found", "department", "patient", "doctor", "bed"]
        result["has_real_data"] = any(indicator in response_lower for indicator in real_data_indicators)
        
        # Print results
        print(f"📊 Search Results: {result['search_results_count']} documents")
        print(f"⏱️  Processing Time: {result['total_time']}s")
        print(f"🤖 AI Model: {result['model_used']}")
        print(f"📋 Document Types: {result['document_types']}")
        
        if expected_keywords:
            print(f"🔑 Keywords Found: {result['keywords_found']}")
        
        print(f"✅ Has Real Data: {'YES' if result['has_real_data'] else 'NO'}")
        print("\n📝 Response:")
        print("-" * 20)
        print(result["response"][:300] + "..." if len(result["response"]) > 300 else result["response"])
        
        self.test_results.append(result)
        return result
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n🚀 Starting Comprehensive RAG Tests")
        print("=" * 60)
        
        tests = [
            {
                "name": "Saudi Patient Count",
                "query": "How many Saudi patients do we have?",
                "keywords": ["54", "saudi", "patients"]
            },
            {
                "name": "Available Beds",
                "query": "How many beds are available in the hospital?",
                "keywords": ["beds", "available", "total"]
            },
            {
                "name": "Orthopedic Specialists",
                "query": "How many doctors are specialized in bones and orthopedics?",
                "keywords": ["doctor", "orthopedic", "bone", "specialist"]
            },
            {
                "name": "Emergency Department Beds",
                "query": "How many beds are available in the Emergency Department?",
                "keywords": ["emergency", "beds", "available"]
            },
            {
                "name": "Cardiology Doctors",
                "query": "Who are the heart specialists in cardiology?",
                "keywords": ["cardiology", "heart", "doctor", "specialist"]
            },
            {
                "name": "Female Patients",
                "query": "How many female patients are currently admitted?",
                "keywords": ["female", "patients", "admitted"]
            },
            {
                "name": "Neurosurgery Specialists",
                "query": "Show me brain surgeons and neurology specialists",
                "keywords": ["brain", "neurosurgery", "neurology", "surgeon"]
            },
            {
                "name": "Pediatric Department",
                "query": "What's the capacity and availability in pediatrics?",
                "keywords": ["pediatric", "capacity", "children", "beds"]
            },
            {
                "name": "Equipment in ICU",
                "query": "What medical equipment is available in the ICU?",
                "keywords": ["equipment", "ICU", "medical", "monitor"]
            },
            {
                "name": "Doctor Availability",
                "query": "Which doctors are currently available for consultation?",
                "keywords": ["doctor", "available", "consultation"]
            },
            {
                "name": "Department Occupancy",
                "query": "Which departments have the highest bed occupancy rates?",
                "keywords": ["department", "occupancy", "beds", "rate"]
            },
            {
                "name": "Surgical Equipment",
                "query": "What surgical equipment needs maintenance?",
                "keywords": ["surgical", "equipment", "maintenance", "surgery"]
            },
            {
                "name": "Patient Blood Types",
                "query": "How many patients have O+ blood type?",
                "keywords": ["blood type", "O+", "patients"]
            },
            {
                "name": "Emergency Cases",
                "query": "How many emergency patients are currently being treated?",
                "keywords": ["emergency", "patients", "treated", "current"]
            },
            {
                "name": "Department Budgets",
                "query": "Which department has the highest annual budget?",
                "keywords": ["department", "budget", "annual", "highest"]
            }
        ]
        
        for i, test in enumerate(tests, 1):
            print(f"\n{'='*10} TEST {i}/{len(tests)} {'='*10}")
            await self.run_test(
                test["name"], 
                test["query"], 
                test.get("keywords", [])
            )
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_searches = sum(1 for r in self.test_results if r["search_results_count"] > 0)
        has_real_data = sum(1 for r in self.test_results if r["has_real_data"])
        ai_successes = sum(1 for r in self.test_results if r["ai_success"])
        
        print(f"🧪 Total Tests Run: {total_tests}")
        print(f"🔍 Successful Searches: {successful_searches}/{total_tests} ({successful_searches/total_tests*100:.1f}%)")
        print(f"📊 Real Data Responses: {has_real_data}/{total_tests} ({has_real_data/total_tests*100:.1f}%)")
        print(f"🤖 AI Model Success: {ai_successes}/{total_tests} ({ai_successes/total_tests*100:.1f}%)")
        
        # Average response time
        avg_time = sum(r["total_time"] for r in self.test_results) / total_tests
        print(f"⏱️  Average Response Time: {avg_time:.3f}s")
        
        # Document type distribution
        all_doc_types = {}
        for result in self.test_results:
            for doc_type, count in result["document_types"].items():
                all_doc_types[doc_type] = all_doc_types.get(doc_type, 0) + count
        
        print(f"\n📋 Document Types Retrieved:")
        for doc_type, count in sorted(all_doc_types.items()):
            print(f"   - {doc_type}: {count} documents")
        
        # Failed tests
        failed_tests = [r for r in self.test_results if r["search_results_count"] == 0]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   - {test['test_name']}: {test['query']}")
        
        # Mock detection
        mock_responses = [r for r in self.test_results if not r["has_real_data"]]
        if mock_responses:
            print(f"\n⚠️  Potential Mock Responses ({len(mock_responses)}):")
            for test in mock_responses:
                print(f"   - {test['test_name']}")
        else:
            print(f"\n✅ NO MOCK RESPONSES DETECTED - All responses use real hospital data!")
        
        # Overall assessment
        print(f"\n{'='*20} FINAL ASSESSMENT {'='*20}")
        
        if has_real_data == total_tests and successful_searches >= total_tests * 0.8:
            print("🎉 FULLY FUNCTIONAL RAG SYSTEM CONFIRMED!")
            print("✅ All responses use real hospital data")
            print("✅ Search functionality works correctly")
            print("✅ No mock or fake responses detected")
        elif has_real_data >= total_tests * 0.8:
            print("✅ RAG SYSTEM IS FUNCTIONAL")
            print(f"✅ {has_real_data}/{total_tests} responses use real data")
            print("⚠️  Some searches may need optimization")
        else:
            print("❌ RAG SYSTEM NEEDS IMPROVEMENT")
            print(f"❌ Only {has_real_data}/{total_tests} responses use real data")
            print("❌ Possible mock responses detected")
        
        print("="*60)

async def main():
    """Main test runner"""
    tester = RAGTester()
    tester.initialize()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())