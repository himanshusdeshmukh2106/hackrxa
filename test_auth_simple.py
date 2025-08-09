#!/usr/bin/env python3
"""
Simple authentication test without document processing
"""
import requests
import json

# Test authentication with invalid data to see if auth works
url = "http://localhost:8000/hackrx/run"
headers = {
    "Authorization": "Bearer 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69",
    "Content-Type": "application/json"
}

# Test 1: Missing required fields (should get 422 validation error, not 401 auth error)
print("=== Test 1: Missing required fields ===")
try:
    response = requests.post(url, headers=headers, json={}, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 401:
        print("❌ Authentication failed")
    elif response.status_code == 422:
        print("✅ Authentication working - got validation error as expected")
    else:
        print(f"⚠️ Unexpected status: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Invalid bearer token (should get 401)
print("=== Test 2: Invalid bearer token ===")
invalid_headers = {
    "Authorization": "Bearer invalid_token",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://example.com/test.pdf",
    "questions": ["What is this?"]
}

try:
    response = requests.post(url, headers=invalid_headers, json=data, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 401:
        print("✅ Authentication properly rejecting invalid tokens")
    else:
        print(f"⚠️ Expected 401, got {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50 + "\n")

# Test 3: No authorization header (should get 401)
print("=== Test 3: No authorization header ===")
no_auth_headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, headers=no_auth_headers, json=data, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 401:
        print("✅ Authentication properly requiring auth header")
    else:
        print(f"⚠️ Expected 401, got {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")