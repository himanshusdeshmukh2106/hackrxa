#!/usr/bin/env python3
"""
Quick test with a very short timeout to see if the fix works
"""
import requests
import json

# Document URL
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

def quick_test():
    """Quick test with minimal timeout"""
    print("⚡ QUICK FIX TEST")
    print("=" * 30)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": ["What is the sum insured?"]
    }
    
    try:
        print("📤 Testing with 30s timeout...")
        response = requests.post(
            "http://127.0.0.1:8000/hackrx/run", 
            headers=headers, 
            json=request_data, 
            timeout=30
        )
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            
            if answers:
                answer = answers[0]
                print(f"📝 Answer: {answer[:150]}...")
                
                # Check if it's still generic
                if "don't have sufficient information" in answer.lower():
                    print("❌ Still generic response")
                    return False
                else:
                    print("✅ Got specific answer!")
                    return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (30s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 SUCCESS: The fix worked!")
        print("🚀 Now run the full performance test")
    else:
        print("\n🔧 Still needs more fixes")
        print("💡 Try restarting the server and test again")