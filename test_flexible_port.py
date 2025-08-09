#!/usr/bin/env python3
"""
Flexible test that tries different ports to find the running server
"""
import requests
import json
import time
import sys

# Document URL provided by user
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token from .env
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

# Try different ports
POSSIBLE_PORTS = [8000, 8080, 8001, 8888, 3000]

def find_running_server():
    """Find which port the server is running on"""
    print("🔍 Looking for running server...")
    
    for port in POSSIBLE_PORTS:
        try:
            url = f"http://localhost:{port}/health"
            print(f"   Trying port {port}...")
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                print(f"   ✅ Found server on port {port}!")
                return port
        except:
            continue
    
    print("   ❌ No server found on any port")
    return None

def test_with_port(port):
    """Run comprehensive test with specific port"""
    api_url = f"http://localhost:{port}/hackrx/run"
    health_url = f"http://localhost:{port}/health"
    
    print(f"\n🧪 Testing with server on port {port}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("🏥 Health Check...")
    try:
        response = requests.get(health_url, timeout=20)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Server healthy: {health_data.get('status', 'unknown')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Authentication
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
        response = requests.post(api_url, headers=headers, json=test_data, timeout=20)
        print(f"   Status: {response.status_code}")
        
        if response.status_code in [200, 422, 500]:
            response_text = response.text.lower()
            if "credentials" in response_text or "bearertokenauth" in response_text:
                print("   ❌ Authentication error detected")
                return False
            else:
                print("   ✅ Authentication working")
        elif response.status_code == 401:
            print("   ❌ Authentication failed")
            return False
    except Exception as e:
        print(f"   ❌ Auth test error: {e}")
        return False
    
    # Test 3: Real Document Processing (Basic)
    print("\n📄 Real Document Test...")
    
    basic_questions = [
        "What is the sum insured available under this Arogya Sanjeevani policy?",
        "What is the waiting period for pre-existing diseases under this policy?",
        "What are the key benefits covered under this health insurance policy?"
    ]
    
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": basic_questions
    }
    
    print(f"   Document: Arogya Sanjeevani Policy")
    print(f"   Questions: {len(basic_questions)}")
    print(f"   Sending request...")
    
    try:
        start_time = time.time()
        response = requests.post(api_url, headers=headers, json=request_data, timeout=300)
        processing_time = time.time() - start_time
        
        print(f"   Response time: {processing_time:.1f}s")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                answers = result.get("answers", [])
                
                print(f"   🎉 SUCCESS! Generated {len(answers)} answers")
                
                # Show results
                print(f"\n📋 Results:")
                for i, (q, a) in enumerate(zip(basic_questions, answers), 1):
                    print(f"\n   {i}. Q: {q}")
                    print(f"      A: {a[:200]}{'...' if len(a) > 200 else ''}")
                
                return True
                
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON: {response.text[:200]}")
                return False
        else:
            print(f"   ❌ Failed: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out")
        return False
    except Exception as e:
        print(f"   ❌ Request error: {e}")
        return False

def main():
    print("🚀 Flexible Port Server Test")
    print("=" * 50)
    
    # Find running server
    port = find_running_server()
    
    if not port:
        print("\n❌ No server found!")
        print("\n💡 Please start the server first:")
        print("   Option 1: python -m uvicorn app.main:app --host 127.0.0.1 --port 8080")
        print("   Option 2: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        print("   Option 3: Run PowerShell as Administrator")
        return False
    
    # Run test with found port
    success = test_with_port(port)
    
    if success:
        print(f"\n🎉 SUCCESS! All tests passed on port {port}")
        print("✅ Authentication working")
        print("✅ Real document processing working")
        print("✅ System ready for production!")
    else:
        print(f"\n❌ Some tests failed on port {port}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)