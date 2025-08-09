#!/usr/bin/env python3
"""
Comprehensive system test using the provided Arogya Sanjeevani Policy document
"""
import requests
import json
import time
from typing import List, Dict

# Document URL provided by user
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token from .env
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

# API endpoint
API_URL = "http://localhost:8000/hackrx/run"

# Test questions about the Arogya Sanjeevani Policy
TEST_QUESTIONS = [
    "What is the sum insured available under this policy?",
    "What is the waiting period for pre-existing diseases?",
    "What are the key benefits covered under this policy?",
    "What is the age limit for entry into this policy?",
    "What is the policy term and premium payment frequency?",
    "Are maternity benefits covered under this policy?",
    "What is the room rent limit under this policy?",
    "What are the exclusions mentioned in this policy?",
    "What is the grace period for premium payment?",
    "What is the free look period for this policy?"
]

def test_server_connectivity():
    """Test basic server connectivity"""
    print("🔍 Testing Server Connectivity")
    print("=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Health endpoint: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   Vector Store: {health_data.get('vector_store', 'unknown')}")
            print(f"   LLM Service: {health_data.get('llm_service', 'unknown')}")
            print("   ✅ Server is responding")
            return True
        else:
            print("   ❌ Health check failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Server connectivity failed: {e}")
        return False

def test_authentication():
    """Test authentication with various scenarios"""
    print("\n🔐 Testing Authentication")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Test with minimal valid request
    test_data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["What is this document about?"]
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=test_data, timeout=10)
        print(f"Authentication test: {response.status_code}")
        
        if response.status_code in [200, 422, 500]:  # 500 might be processing error, not auth error
            response_text = response.text.lower()
            if "credentials" in response_text or "bearertokenauth" in response_text:
                print("   ❌ Authentication error still exists")
                return False
            else:
                print("   ✅ Authentication is working")
                return True
        elif response.status_code == 401:
            print("   ❌ Authentication failed unexpectedly")
            return False
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Authentication test failed: {e}")
        return False

def test_document_processing():
    """Test document processing with the provided Arogya Sanjeevani Policy"""
    print("\n📄 Testing Document Processing")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Start with a subset of questions to test basic functionality
    test_questions = TEST_QUESTIONS[:3]  # First 3 questions
    
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": test_questions
    }
    
    print(f"Testing with document: {DOCUMENT_URL[:80]}...")
    print(f"Questions to ask: {len(test_questions)}")
    
    try:
        print("   Sending request to API...")
        start_time = time.time()
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=request_data, 
            timeout=120  # 2 minutes timeout for document processing
        )
        
        processing_time = time.time() - start_time
        print(f"   Response received in {processing_time:.2f} seconds")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                answers = result.get("answers", [])
                
                print(f"   ✅ Successfully processed document!")
                print(f"   📊 Generated {len(answers)} answers")
                
                # Display results
                print("\n📋 Results:")
                print("-" * 40)
                for i, (question, answer) in enumerate(zip(test_questions, answers), 1):
                    print(f"\n{i}. Q: {question}")
                    print(f"   A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                
                return True, answers
                
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON response: {response.text[:200]}")
                return False, None
                
        elif response.status_code == 422:
            print(f"   ❌ Validation error: {response.text}")
            return False, None
            
        elif response.status_code == 500:
            print(f"   ❌ Server error: {response.text}")
            # Check if it's a specific processing error
            if "document" in response.text.lower():
                print("   💡 This appears to be a document processing issue, not authentication")
            return False, None
            
        else:
            print(f"   ❌ Unexpected status {response.status_code}: {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out (document processing took too long)")
        return False, None
    except Exception as e:
        print(f"   ❌ Document processing failed: {e}")
        return False, None

def test_extended_questions():
    """Test with more questions if basic test passes"""
    print("\n🔍 Testing Extended Question Set")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Use more questions
    extended_questions = TEST_QUESTIONS[:7]  # First 7 questions
    
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": extended_questions
    }
    
    print(f"Testing with {len(extended_questions)} questions...")
    
    try:
        start_time = time.time()
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=request_data, 
            timeout=180  # 3 minutes for extended test
        )
        processing_time = time.time() - start_time
        
        print(f"   Response received in {processing_time:.2f} seconds")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            print(f"   ✅ Successfully processed {len(answers)} questions!")
            print(f"   ⚡ Average time per question: {processing_time/len(extended_questions):.2f}s")
            
            return True, answers
        else:
            print(f"   ❌ Extended test failed: {response.status_code}")
            return False, None
            
    except Exception as e:
        print(f"   ❌ Extended test error: {e}")
        return False, None

def main():
    """Run comprehensive system tests"""
    print("🚀 Comprehensive System Test - Arogya Sanjeevani Policy")
    print("=" * 70)
    print(f"Document: Arogya Sanjeevani Policy")
    print(f"Questions: {len(TEST_QUESTIONS)} prepared")
    print("=" * 70)
    
    # Test 1: Server connectivity
    connectivity_ok = test_server_connectivity()
    
    # Test 2: Authentication
    auth_ok = test_authentication() if connectivity_ok else False
    
    # Test 3: Basic document processing
    basic_processing_ok = False
    basic_answers = None
    if auth_ok:
        basic_processing_ok, basic_answers = test_document_processing()
    
    # Test 4: Extended questions (only if basic test passes)
    extended_processing_ok = False
    extended_answers = None
    if basic_processing_ok:
        extended_processing_ok, extended_answers = test_extended_questions()
    
    # Final Results
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    print(f"🔗 Server Connectivity: {'✅ PASS' if connectivity_ok else '❌ FAIL'}")
    print(f"🔐 Authentication: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"📄 Basic Processing: {'✅ PASS' if basic_processing_ok else '❌ FAIL'}")
    print(f"🔍 Extended Processing: {'✅ PASS' if extended_processing_ok else '❌ FAIL'}")
    
    if basic_processing_ok:
        print(f"\n🎉 SUCCESS: System is working with real document!")
        print(f"   - Document URL accessible and processed")
        print(f"   - Questions answered successfully")
        print(f"   - Authentication working correctly")
        
        if extended_processing_ok:
            print(f"   - Extended question set also working")
            
        print(f"\n💡 The LLM Query Retrieval System is ready for production use!")
        return True
    else:
        print(f"\n❌ ISSUES FOUND:")
        if not connectivity_ok:
            print(f"   - Server connectivity problems")
        if not auth_ok:
            print(f"   - Authentication issues")
        if not basic_processing_ok:
            print(f"   - Document processing problems")
            
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)