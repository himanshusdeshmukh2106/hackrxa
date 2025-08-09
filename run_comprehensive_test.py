#!/usr/bin/env python3
"""
Script to start server and run comprehensive test
"""
import subprocess
import time
import requests
import sys
import os

def wait_for_server(max_wait=60):
    """Wait for server to be ready"""
    print("Waiting for server to start...")
    
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print(f"Server is ready! (took {i+1} seconds)")
                return True
        except:
            pass
        
        print(f"   Waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print("❌ Server failed to start within timeout")
    return False

def main():
    print("Starting Server and Running Comprehensive Test")
    print("=" * 60)
    
    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ Server is already running!")
            server_ready = True
        else:
            server_ready = False
    except:
        server_ready = False
    
    # Start server if not running
    if not server_ready:
        print("🔄 Starting server...")
        
        # Start server in background
        server_process = subprocess.Popen([
            "python", "-m", "uvicorn", "app.main:app", 
            "--host", "0.0.0.0", "--port", "8000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to be ready
        server_ready = wait_for_server(60)
        
        if not server_ready:
            print("❌ Failed to start server")
            server_process.terminate()
            return False
    
    # Run the comprehensive test
    print("\n🧪 Running Comprehensive Test...")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            "python", "test_real_document.py"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 10 minutes")
        success = False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        success = False
    
    # Cleanup
    if 'server_process' in locals():
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)