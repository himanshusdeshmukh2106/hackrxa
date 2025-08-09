#!/usr/bin/env python3
"""
Direct test of search engine to identify the issue
"""
import requests
import json

# Document URL
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

def test_single_question():
    """Test with a single question to see detailed response"""
    print("🔍 TESTING SINGLE QUESTION")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Test with a very specific question that should have clear answers
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": ["What is the sum insured amount mentioned in this policy?"]
    }
    
    try:
        print("📤 Making API request...")
        print(f"   Question: {request_data['questions'][0]}")
        
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
                answer = answers[0]
                print(f"\n📝 FULL ANSWER:")
                print(f"   {answer}")
                
                # Check if it contains actual policy information
                policy_keywords = [
                    "sum insured", "amount", "rupees", "rs", "lakh", "crore",
                    "premium", "coverage", "benefit", "policy", "insurance"
                ]
                
                found_keywords = [kw for kw in policy_keywords if kw.lower() in answer.lower()]
                print(f"\n🔍 Policy keywords found: {found_keywords}")
                
                if found_keywords:
                    print("   ✅ Answer contains policy-related information!")
                else:
                    print("   ❌ Answer is generic - no policy information found")
                    
                    # Check if it's the standard "no information" response
                    if "don't have sufficient information" in answer.lower():
                        print("   🚨 This is the standard 'no information' response")
                        print("   🔧 The search engine is not finding relevant content")
            
            # Check for any additional metadata
            if "processing_time" in result:
                print(f"\n⏱️ Processing time: {result['processing_time']}s")
            
            if "document_chunks" in result:
                print(f"📊 Document chunks processed: {result['document_chunks']}")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_health_check():
    """Check if server is healthy"""
    print("\n🏥 HEALTH CHECK")
    print("=" * 30)
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Vector store: {health_data.get('vector_store', 'Unknown')}")
            print(f"   LLM service: {health_data.get('llm_service', 'Unknown')}")
            print(f"   Embedding service: {health_data.get('embedding_service', 'Unknown')}")
        
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")

if __name__ == "__main__":
    print("🧪 DIRECT SEARCH ENGINE TEST")
    print("This will test a single question to see the exact response")
    print()
    
    test_health_check()
    test_single_question()
    
    print("\n" + "=" * 50)
    print("🔧 If the answer is still generic, the issue is:")
    print("   1. Search engine threshold still too high")
    print("   2. Document ID mismatch in search filters")
    print("   3. Search engine not properly initialized")
    print("   4. LLM not receiving the found content")