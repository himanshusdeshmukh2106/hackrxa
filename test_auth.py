#!/usr/bin/env python3
"""
Quick test script to check if authentication is working
"""
import requests
import json

# Test data
url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}

try:
    print("Testing authentication...")
    response = requests.post(url, headers=headers, json=data, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 500:
        print("❌ Still getting 500 error - authentication issue not fixed")
    elif response.status_code == 422:
        print("✅ Authentication working - got validation error (expected)")
    elif response.status_code == 200:
        print("✅ Authentication and processing working!")
    else:
        print(f"⚠️ Unexpected status code: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server - make sure it's running on localhost:8000")
except Exception as e:
    print(f"❌ Error: {e}")