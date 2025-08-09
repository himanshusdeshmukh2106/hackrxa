#!/usr/bin/env python3
"""
Debug script to check document processing pipeline
"""
import requests
import json

# Document URL
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

def test_document_access():
    """Test if we can access the document directly"""
    print("🔍 Testing document access...")
    
    try:
        response = requests.head(DOCUMENT_URL, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        print(f"   Content-Length: {response.headers.get('Content-Length', 'Unknown')}")
        
        if response.status_code == 200:
            print("   ✅ Document is accessible")
            return True
        else:
            print("   ❌ Document access failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error accessing document: {e}")
        return False

def test_api_with_debug():
    """Test API with detailed debugging"""
    print("\n🔍 Testing API with debug info...")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Simple test with one question
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": ["What is the sum insured under this policy?"]
    }
    
    try:
        print("   Making API request...")
        response = requests.post(
            "http://127.0.0.1:8000/hackrx/run", 
            headers=headers, 
            json=request_data, 
            timeout=60
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Response keys: {list(result.keys())}")
            
            answers = result.get("answers", [])
            print(f"   Number of answers: {len(answers)}")
            
            if answers:
                print(f"   First answer: {answers[0]}")
                print(f"   Answer length: {len(answers[0])}")
                
                # Check if it's a generic "no info" response
                no_info_indicators = [
                    "don't have sufficient information",
                    "not provided in the document",
                    "cannot find",
                    "no information available"
                ]
                
                is_generic = any(indicator in answers[0].lower() for indicator in no_info_indicators)
                print(f"   Is generic response: {is_generic}")
            
            # Check for processing metadata
            if "processing_time" in result:
                print(f"   Processing time: {result['processing_time']}s")
            
            if "document_chunks" in result:
                print(f"   Document chunks: {result['document_chunks']}")
                
        else:
            print(f"   Error response: {response.text[:500]}")
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")

def check_server_logs():
    """Check if server is running and accessible"""
    print("\n🔍 Checking server health...")
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"   Health check status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Health data: {json.dumps(health_data, indent=2)}")
        
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")

if __name__ == "__main__":
    print("🔧 DOCUMENT PROCESSING DEBUG")
    print("=" * 50)
    
    # Test document access
    doc_accessible = test_document_access()
    
    # Test server health
    check_server_logs()
    
    # Test API
    if doc_accessible:
        test_api_with_debug()
    else:
        print("\n⚠️ Skipping API test - document not accessible")
    
    print("\n" + "=" * 50)
    print("🔧 Debug complete. Check the output above for issues.")