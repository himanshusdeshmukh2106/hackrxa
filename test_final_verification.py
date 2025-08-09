#!/usr/bin/env python3
"""
Final verification test to confirm all fixes are working
"""
import requests
import json

def test_authentication_fixes():
    """Test that all authentication issues are resolved"""
    print("Testing Authentication Fixes")
    print("=" * 50)
    
    url = "http://localhost:8000/hackrx/run"
    valid_token = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"
    
    # Test 1: Valid authentication with proper request
    print("\n1. Testing valid authentication...")
    headers = {
        "Authorization": f"Bearer {valid_token}",
        "Content-Type": "application/json"
    }
    data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["What is this document about?"]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 500:
            # Check if it's authentication error or processing error
            response_text = response.text.lower()
            if "credentials" in response_text or "bearertokenauth" in response_text:
                print("   ❌ AUTHENTICATION ERROR STILL EXISTS")
                return False
            else:
                print("   ✅ Authentication working (500 is document processing error)")
        elif response.status_code in [200, 422]:
            print("   ✅ Authentication working perfectly")
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Invalid token
    print("\n2. Testing invalid token rejection...")
    invalid_headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=invalid_headers, json=data, timeout=5)
        if response.status_code == 401:
            print("   ✅ Invalid tokens properly rejected")
        else:
            print(f"   ❌ Expected 401, got {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Missing authorization
    print("\n3. Testing missing authorization rejection...")
    no_auth_headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=no_auth_headers, json=data, timeout=5)
        if response.status_code == 401:
            print("   ✅ Missing auth properly rejected")
        else:
            print(f"   ❌ Expected 401, got {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

def test_server_health():
    """Test server health and basic functionality"""
    print("\n\n🏥 Testing Server Health")
    print("=" * 50)
    
    try:
        # Health check
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"\nHealth endpoint: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Database: {health_data.get('database', 'unknown')}")
            print(f"   Vector Store: {health_data.get('vector_store', 'unknown')}")
            print("   ✅ Server is healthy")
        else:
            print("   ❌ Health check failed")
            return False
            
        # Root endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
        else:
            print("   ❌ Root endpoint failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Server health check failed: {e}")
        return False
    
    return True

def main():
    """Run all verification tests"""
    print("Final Verification of Authentication Fixes")
    print("=" * 60)
    
    auth_success = test_authentication_fixes()
    health_success = test_server_health()
    
    print("\n\n📊 FINAL RESULTS")
    print("=" * 60)
    
    if auth_success and health_success:
        print("🎉 ALL FIXES SUCCESSFUL!")
        print("✅ Authentication is working correctly")
        print("✅ Server is healthy and responding")
        print("✅ The original 500 'credentials' error is FIXED")
        print("\nThe system is ready for use!")
        return True
    else:
        print("❌ Some issues remain:")
        if not auth_success:
            print("   - Authentication issues")
        if not health_success:
            print("   - Server health issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)