#!/usr/bin/env python3
"""
Direct test for server on port 8000
"""
import requests
import json
import time
import sys

# Document URL provided by user
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token from .env
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

# Direct URLs for port 8000
API_URL = "http://127.0.0.1:8000/hackrx/run"
HEALTH_URL = "http://127.0.0.1:8000/health"

def main():
    print("🚀 Direct Test - Port 8000")
    print("=" * 50)
    
    # Test 1: Health Check with very long timeout
    print("🏥 Health Check (this may take 30+ seconds)...")
    try:
        print("   Sending health request...")
        response = requests.get(HEALTH_URL, timeout=60)  # 1 minute timeout
        print(f"   ✅ Health response: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Server status: {health_data.get('status', 'unknown')}")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   Vector Store: {health_data.get('vector_store', 'unknown')}")
            print(f"   LLM Service: {health_data.get('llm_service', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Health check timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Authentication Test
    print("\n🔐 Authentication Test...")
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["What is this document about?"]
    }
    
    try:
        print("   Testing authentication...")
        response = requests.post(API_URL, headers=headers, json=test_data, timeout=30)
        print(f"   Auth response: {response.status_code}")
        
        if response.status_code in [200, 422, 500]:
            response_text = response.text.lower()
            if "credentials" in response_text or "bearertokenauth" in response_text:
                print("   ❌ Authentication error detected!")
                print(f"   Response: {response.text[:200]}")
                return False
            else:
                print("   ✅ Authentication working (no credential errors)")
        elif response.status_code == 401:
            print("   ❌ Authentication failed")
            print(f"   Response: {response.text}")
            return False
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Auth test error: {e}")
        return False
    
    # Test 3: Simple Real Document Test
    print("\n📄 Real Document Test (1 question)...")
    
    simple_request = {
        "documents": DOCUMENT_URL,
        "questions": ["What is the sum insured available under this Arogya Sanjeevani policy?"]
    }
    
    print("   Document: Arogya Sanjeevani Policy")
    print("   Question: Sum insured query")
    print("   Sending request (this may take 2-3 minutes)...")
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, json=simple_request, timeout=300)  # 5 minutes
        processing_time = time.time() - start_time
        
        print(f"   Response time: {processing_time:.1f} seconds")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                answers = result.get("answers", [])
                
                if answers:
                    print(f"   🎉 SUCCESS! Generated answer:")
                    print(f"   Q: What is the sum insured available under this Arogya Sanjeevani policy?")
                    print(f"   A: {answers[0]}")
                    return True
                else:
                    print("   ❌ No answers received")
                    return False
                    
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON response")
                print(f"   Response: {response.text[:300]}")
                return False
                
        elif response.status_code == 422:
            print(f"   ❌ Validation error: {response.text}")
            return False
        elif response.status_code == 500:
            print(f"   ❌ Server error: {response.text}")
            return False
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"   ❌ Request error: {e}")
        return False

if __name__ == "__main__":
    print("⏰ Note: This test may take several minutes due to server startup and document processing")
    print("Please be patient...\n")
    
    success = main()
    
    if success:
        print(f"\n🎉 SUCCESS! All tests passed!")
        print("✅ Server is running and healthy")
        print("✅ Authentication is working correctly")
        print("✅ Real document processing is working")
        print("✅ The LLM Query Retrieval System is fully functional!")
    else:
        print(f"\n❌ Some tests failed")
        print("Please check the server logs for more details")
    
    sys.exit(0 if success else 1)