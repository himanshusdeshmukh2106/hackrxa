#!/usr/bin/env python3
"""
System validation script for LLM-Powered Query Retrieval System
This script validates that all components are working correctly
"""
import asyncio
import sys
import time
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str = "INFO"):
    """Print colored status message"""
    color = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED
    }.get(status, Colors.BLUE)
    
    print(f"{color}[{status}]{Colors.END} {message}")

def print_header(message: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

class SystemValidator:
    """Validates the complete LLM Query Retrieval System"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        
        self.validation_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "warnings": 0,
            "errors": []
        }
    
    def validate_all(self) -> bool:
        """Run all validation tests"""
        print_header("LLM Query Retrieval System Validation")
        print(f"Testing system at: {self.base_url}")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        
        # Run all validation tests
        tests = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("Health Check", self.test_health_endpoint),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("API Documentation", self.test_api_documentation),
            ("Authentication", self.test_authentication),
            ("Request Validation", self.test_request_validation),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance),
            ("Rate Limiting", self.test_rate_limiting),
            ("System Integration", self.test_system_integration)
        ]
        
        for test_name, test_func in tests:
            print_header(f"Testing: {test_name}")
            try:
                test_func()
            except Exception as e:
                self.record_error(test_name, str(e))
                print_status(f"Test failed with exception: {str(e)}", "ERROR")
        
        # Print summary
        self.print_summary()
        
        return self.validation_results["failed_tests"] == 0
    
    def record_test(self, passed: bool, message: str = ""):
        """Record test result"""
        self.validation_results["total_tests"] += 1
        if passed:
            self.validation_results["passed_tests"] += 1
            print_status(f"✓ {message}", "SUCCESS")
        else:
            self.validation_results["failed_tests"] += 1
            print_status(f"✗ {message}", "ERROR")
    
    def record_warning(self, message: str):
        """Record warning"""
        self.validation_results["warnings"] += 1
        print_status(f"⚠ {message}", "WARNING")
    
    def record_error(self, test_name: str, error: str):
        """Record error"""
        self.validation_results["errors"].append(f"{test_name}: {error}")
        self.validation_results["failed_tests"] += 1
    
    def test_basic_connectivity(self):
        """Test basic connectivity to the system"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            self.record_test(
                response.status_code == 200,
                f"Basic connectivity (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Basic connectivity failed: {str(e)}")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            self.record_test(
                response.status_code == 200,
                f"Health endpoint accessibility (status: {response.status_code})"
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["status", "timestamp", "version"]
                for field in required_fields:
                    self.record_test(
                        field in data,
                        f"Health response contains '{field}'"
                    )
                
                # Check status value
                if "status" in data:
                    status = data["status"]
                    if status == "healthy":
                        print_status("System status: healthy", "SUCCESS")
                    elif status == "degraded":
                        self.record_warning("System status: degraded")
                    else:
                        self.record_test(False, f"System status: {status}")
                
                # Check service health
                services = ["database", "vector_store", "llm_service"]
                for service in services:
                    if service in data:
                        service_status = data[service]
                        self.record_test(
                            service_status is True,
                            f"{service} health: {'healthy' if service_status else 'unhealthy'}"
                        )
        
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Health endpoint failed: {str(e)}")
        except json.JSONDecodeError:
            self.record_test(False, "Health endpoint returned invalid JSON")
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics", headers=self.headers, timeout=10)
            
            if response.status_code == 401 and not self.token:
                self.record_warning("Metrics endpoint requires authentication (expected)")
                return
            
            self.record_test(
                response.status_code == 200,
                f"Metrics endpoint accessibility (status: {response.status_code})"
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["system_status", "cache_stats", "timestamp"]
                for field in required_fields:
                    self.record_test(
                        field in data,
                        f"Metrics response contains '{field}'"
                    )
        
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Metrics endpoint failed: {str(e)}")
        except json.JSONDecodeError:
            self.record_test(False, "Metrics endpoint returned invalid JSON")
    
    def test_api_documentation(self):
        """Test API documentation endpoints"""
        endpoints = [
            ("/docs", "Swagger UI"),
            ("/redoc", "ReDoc"),
            ("/openapi.json", "OpenAPI JSON")
        ]
        
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                self.record_test(
                    response.status_code == 200,
                    f"{name} accessible (status: {response.status_code})"
                )
            except requests.exceptions.RequestException as e:
                self.record_test(False, f"{name} failed: {str(e)}")
    
    def test_authentication(self):
        """Test authentication requirements"""
        # Test without authentication
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json={"documents": "https://example.com/test.pdf", "questions": ["test"]},
                timeout=10
            )
            self.record_test(
                response.status_code == 401,
                f"Authentication required (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Authentication test failed: {str(e)}")
        
        # Test with invalid token
        try:
            invalid_headers = {"Authorization": "Bearer invalid_token"}
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json={"documents": "https://example.com/test.pdf", "questions": ["test"]},
                headers=invalid_headers,
                timeout=10
            )
            self.record_test(
                response.status_code == 401,
                f"Invalid token rejected (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Invalid token test failed: {str(e)}")
        
        # Test with valid token (if provided)
        if self.token:
            try:
                response = requests.post(
                    f"{self.base_url}/hackrx/run",
                    json={"documents": "https://example.com/test.pdf", "questions": ["test"]},
                    headers=self.headers,
                    timeout=30
                )
                # Should not be 401 with valid token
                self.record_test(
                    response.status_code != 401,
                    f"Valid token accepted (status: {response.status_code})"
                )
            except requests.exceptions.RequestException as e:
                self.record_warning(f"Valid token test failed: {str(e)}")
    
    def test_request_validation(self):
        """Test request validation"""
        if not self.token:
            self.record_warning("Skipping request validation tests (no token provided)")
            return
        
        # Test missing documents
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json={"questions": ["test"]},
                headers=self.headers,
                timeout=10
            )
            self.record_test(
                response.status_code == 422,
                f"Missing documents rejected (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Missing documents test failed: {str(e)}")
        
        # Test missing questions
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json={"documents": "https://example.com/test.pdf"},
                headers=self.headers,
                timeout=10
            )
            self.record_test(
                response.status_code == 422,
                f"Missing questions rejected (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Missing questions test failed: {str(e)}")
        
        # Test empty questions
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json={"documents": "https://example.com/test.pdf", "questions": []},
                headers=self.headers,
                timeout=10
            )
            self.record_test(
                response.status_code == 422,
                f"Empty questions rejected (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Empty questions test failed: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling"""
        # Test invalid JSON
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                data="invalid json",
                headers={**self.headers, "Content-Type": "application/json"},
                timeout=10
            )
            self.record_test(
                response.status_code in [400, 422],
                f"Invalid JSON rejected (status: {response.status_code})"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Invalid JSON test failed: {str(e)}")
        
        # Test error response format
        try:
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json={"invalid": "request"},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    # Check error response format
                    if isinstance(error_data, dict) and "detail" in error_data:
                        # FastAPI format
                        self.record_test(True, "Error response format valid (FastAPI)")
                    elif isinstance(error_data, dict) and "error" in error_data:
                        # Custom format
                        self.record_test(True, "Error response format valid (Custom)")
                    else:
                        self.record_test(False, "Error response format invalid")
                except json.JSONDecodeError:
                    self.record_test(False, "Error response not valid JSON")
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Error format test failed: {str(e)}")
    
    def test_performance(self):
        """Test performance characteristics"""
        if not self.token:
            self.record_warning("Skipping performance tests (no token provided)")
            return
        
        # Test response time for health endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            self.record_test(
                response_time < 5.0,
                f"Health endpoint response time: {response_time:.2f}s"
            )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Health performance test failed: {str(e)}")
        
        # Test response time for metrics endpoint
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/metrics", headers=self.headers, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                self.record_test(
                    response_time < 5.0,
                    f"Metrics endpoint response time: {response_time:.2f}s"
                )
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Metrics performance test failed: {str(e)}")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        if not self.token:
            self.record_warning("Skipping rate limiting tests (no token provided)")
            return
        
        # Make multiple requests to test rate limiting headers
        try:
            response = requests.get(f"{self.base_url}/metrics", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                # Check for rate limiting headers
                rate_limit_headers = [
                    "X-RateLimit-Limit",
                    "X-RateLimit-Remaining",
                    "X-RateLimit-Reset"
                ]
                
                found_headers = []
                for header in rate_limit_headers:
                    if header in response.headers:
                        found_headers.append(header)
                
                if found_headers:
                    self.record_test(True, f"Rate limiting headers present: {', '.join(found_headers)}")
                else:
                    self.record_warning("No rate limiting headers found")
        
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Rate limiting test failed: {str(e)}")
    
    def test_system_integration(self):
        """Test system integration with a simple query"""
        if not self.token:
            self.record_warning("Skipping integration tests (no token provided)")
            return
        
        # Test with a simple request
        test_request = {
            "documents": "https://example.com/test-document.pdf",
            "questions": ["What is the main topic of this document?"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/hackrx/run",
                json=test_request,
                headers=self.headers,
                timeout=35  # Allow extra time for processing
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Check response status
            if response.status_code == 200:
                self.record_test(True, f"Integration test successful (time: {response_time:.2f}s)")
                
                # Check response format
                try:
                    data = response.json()
                    if "answers" in data and isinstance(data["answers"], list):
                        self.record_test(True, "Response format valid")
                        
                        if len(data["answers"]) == 1:
                            self.record_test(True, "Correct number of answers returned")
                        else:
                            self.record_test(False, f"Expected 1 answer, got {len(data['answers'])}")
                    else:
                        self.record_test(False, "Invalid response format")
                except json.JSONDecodeError:
                    self.record_test(False, "Response not valid JSON")
                
                # Check response time
                self.record_test(
                    response_time < 30.0,
                    f"Response time under 30s: {response_time:.2f}s"
                )
            
            elif response.status_code == 422:
                # Document processing error is expected for test URL
                self.record_warning("Document processing failed (expected for test URL)")
            
            elif response.status_code == 503:
                self.record_warning("External services unavailable")
            
            else:
                self.record_test(False, f"Integration test failed (status: {response.status_code})")
        
        except requests.exceptions.Timeout:
            self.record_test(False, "Integration test timed out")
        except requests.exceptions.RequestException as e:
            self.record_test(False, f"Integration test failed: {str(e)}")
    
    def print_summary(self):
        """Print validation summary"""
        print_header("Validation Summary")
        
        results = self.validation_results
        
        print(f"Total Tests: {results['total_tests']}")
        print_status(f"Passed: {results['passed_tests']}", "SUCCESS")
        
        if results['failed_tests'] > 0:
            print_status(f"Failed: {results['failed_tests']}", "ERROR")
        
        if results['warnings'] > 0:
            print_status(f"Warnings: {results['warnings']}", "WARNING")
        
        if results['errors']:
            print("\nErrors encountered:")
            for error in results['errors']:
                print_status(f"  {error}", "ERROR")
        
        # Overall status
        if results['failed_tests'] == 0:
            print_status("\n🎉 System validation PASSED!", "SUCCESS")
        else:
            print_status(f"\n❌ System validation FAILED ({results['failed_tests']} failures)", "ERROR")
        
        # Success rate
        if results['total_tests'] > 0:
            success_rate = (results['passed_tests'] / results['total_tests']) * 100
            print(f"\nSuccess Rate: {success_rate:.1f}%")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate LLM Query Retrieval System")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the system")
    parser.add_argument("--token", help="Bearer token for authentication")
    parser.add_argument("--config", help="Configuration file with URL and token")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                args.url = config.get('url', args.url)
                args.token = config.get('token', args.token)
        except Exception as e:
            print_status(f"Failed to load config file: {str(e)}", "ERROR")
            sys.exit(1)
    
    # Create validator and run tests
    validator = SystemValidator(args.url, args.token)
    success = validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()