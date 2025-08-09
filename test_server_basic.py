#!/usr/bin/env python3
"""
Basic server test to check if it's responsive
"""
import requests

def test_basic_endpoints():
    """Test basic server endpoints"""
    print("🔍 BASIC SERVER TEST")
    print("=" * 30)
    
    # Test root endpoint
    try:
        print("📡 Testing root endpoint...")
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        print(f"   Root: {response.status_code}")
    except Exception as e:
        print(f"   Root failed: {e}")
    
    # Test health endpoint
    try:
        print("🏥 Testing health endpoint...")
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        print(f"   Health: {response.status_code}")
        if response.status_code == 200:
            print(f"   Health data: {response.json()}")
    except Exception as e:
        print(f"   Health failed: {e}")
    
    # Test docs endpoint
    try:
        print("📚 Testing docs endpoint...")
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        print(f"   Docs: {response.status_code}")
    except Exception as e:
        print(f"   Docs failed: {e}")

if __name__ == "__main__":
    test_basic_endpoints()