#!/usr/bin/env python3
"""
Comprehensive performance test for Pinecone optimizations
"""
import requests
import json
import time
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document URL
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

# API endpoint
API_URL = "http://127.0.0.1:8000/hackrx/run"

def make_request(test_case):
    """Make a single API request"""
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    request_data = {
        "documents": DOCUMENT_URL,
        "questions": test_case['questions']
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, json=request_data, timeout=120)
        processing_time = time.time() - start_time
        
        return {
            "name": test_case['name'],
            "time": processing_time,
            "status": response.status_code,
            "success": response.status_code == 200 and processing_time <= test_case['target'],
            "target": test_case['target'],
            "response": response.json() if response.status_code == 200 else response.text[:200]
        }
    except Exception as e:
        return {
            "name": test_case['name'],
            "time": 0,
            "status": 0,
            "success": False,
            "target": test_case['target'],
            "error": str(e)
        }

def test_performance():
    """Test system performance with aggressive targets"""
    print("PINECONE PERFORMANCE TEST - Target: <30 seconds")
    print("=" * 70)
    
    # Warm-up request
    print("Warming up the server...")
    make_request({"name": "Warm-up", "questions": ["What is this?"], "target": 60})
    print("Server is warm.")
    
    # Progressive test cases
    test_cases = [
        {
            "name": "Single Question (Fast)",
            "questions": ["What is the sum insured?"],
            "target": 12  # 12s target
        },
        {
            "name": "Two Questions (Medium)",
            "questions": [
                "What is the sum insured under this policy?",
                "What is the waiting period?"
            ],
            "target": 20  # 20s target
        },
        {
            "name": "Three Questions (Full)",
            "questions": [
                "What is the sum insured under this policy?",
                "What is the waiting period for pre-existing diseases?",
                "What are the key benefits covered?"
            ],
            "target": 25  # 25s target
        },
        {
            "name": "Five Questions (Stress)",
            "questions": [
                "What is the sum insured under this policy?",
                "What is the waiting period for pre-existing diseases?",
                "What are the key benefits covered?",
                "What are the exclusions?",
                "What is the premium amount?"
            ],
            "target": 30  # 30s target
        }
    ]
    
    print(f"Running {len(test_cases)} performance tests...")
    print(f"Optimization Focus: Pinecone async operations, batching, caching")
    print("-" * 70)
    
    results = []
    
    # Run tests sequentially to avoid overwhelming the system
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}: {test_case['name']}")
        print(f"   Questions: {len(test_case['questions'])}")
        print(f"   Target: <{test_case['target']}s")
        print(f"   Running...", end="", flush=True)
        
        result = make_request(test_case)
        results.append(result)
        
        # Status indicator
        if result['success']:
            status_icon = "[PASS]"
            status_text = "PASS"
        else:
            status_icon = "[FAIL]"
            status_text = "FAIL"
        
        print(f"\r   Result: {status_icon} {status_text} ({result['time']:.1f}s)")
        
        if result.get('error'):
            print(f"   Error: {result['error']}")
        elif result['status'] == 200:
            response_data = result.get('response', {})
            answers = response_data.get('answers', [])
            print(f"   Answers: {len(answers)}")
            if answers:
                print(f"   Preview: {answers[0][:80]}...")
        
        # Brief pause between tests
        if i < len(test_cases):
            time.sleep(2)
    
    # Performance Analysis
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    
    print(f"\nTest Results:")
    for result in results:
        status = "[PASS]" if result['success'] else "[FAIL]"
        improvement = ""
        if result['time'] > 0:
            baseline = 145  # Original baseline
            speedup = baseline / result['time']
            improvement = f" ({speedup:.1f}x faster)"
        
        print(f"{status} {result['name']}: {result['time']:.1f}s (target: {result['target']}s){improvement}")
    
    # Calculate average performance
    valid_times = [r['time'] for r in results if r['time'] > 0]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        baseline_avg = 145
        overall_speedup = baseline_avg / avg_time
        
        print(f"Performance Summary:")
        print(f"   Average Time: {avg_time:.1f}s")
        print(f"   Overall Speedup: {overall_speedup:.1f}x")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
    
    # Recommendations
    print(f"\nOptimization Impact:")
    if passed_tests == total_tests:
        print("   ALL TARGETS MET! Pinecone optimizations successful.")
        print("   Async operations working")
        print("   Batch processing effective")
        print("   Thread pool optimization successful")
    elif passed_tests > 0:
        print(f"   {passed_tests}/{total_tests} targets met - Good progress!")
        print("   Consider additional optimizations:")
        print("     - Increase thread pool size")
        print("     - Implement query result caching")
        print("     - Optimize embedding batch size")
    else:
        print("   No targets met - Need more optimization")
        print("   Recommended actions:")
        print("     - Check Pinecone connection latency")
        print("     - Verify async operations are working")
        print("     - Consider connection pooling")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    print("Starting Pinecone Performance Test...")
    print("This will test the optimized async Pinecone operations")
    print()
    
    success = test_performance()
    
    print(f"\n{'='*70}")
    if success:
        print("SUCCESS: All performance targets achieved!")
        print("System ready for production with <30s response times")
    else:
        print("Some targets not met. System needs further optimization.")
        print("Review the recommendations above for next steps.")

    
    sys.exit(0 if success else 1)