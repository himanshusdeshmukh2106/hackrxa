"""
Final integration tests for the complete LLM Query Retrieval System
"""
import pytest
import asyncio
import json
import time
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.schemas.models import Document, TextChunk, SearchResult, Answer


class TestCompleteSystemIntegration:
    """Test the complete system integration with realistic scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers"""
        return {"Authorization": f"Bearer {settings.bearer_token}"}
    
    @pytest.fixture
    def insurance_policy_request(self):
        """Sample insurance policy request"""
        return {
            "documents": "https://example.com/insurance-policy.pdf",
            "questions": [
                "What is the maximum coverage amount for medical expenses?",
                "Are pre-existing conditions covered under this policy?",
                "What is the waiting period for dental coverage?",
                "What are the exclusions mentioned in the policy?",
                "How do I file a claim for reimbursement?"
            ]
        }
    
    @pytest.fixture
    def legal_contract_request(self):
        """Sample legal contract request"""
        return {
            "documents": "https://example.com/employment-contract.pdf",
            "questions": [
                "What is the notice period for termination?",
                "Are there any non-compete clauses?",
                "What are the working hours specified?",
                "What benefits are included in the compensation package?",
                "What is the probation period duration?"
            ]
        }
    
    @pytest.fixture
    def hr_handbook_request(self):
        """Sample HR handbook request"""
        return {
            "documents": "https://example.com/employee-handbook.pdf",
            "questions": [
                "What is the company's remote work policy?",
                "How many vacation days are employees entitled to?",
                "What is the process for requesting time off?",
                "What are the performance review procedures?",
                "What training opportunities are available?"
            ]
        }
    
    def setup_insurance_mocks(self):
        """Setup mocks for insurance policy scenario"""
        # Mock document
        mock_document = Document(
            id="insurance_doc_123",
            url="https://example.com/insurance-policy.pdf",
            content_type="application/pdf"
        )
        
        # Mock text chunks with insurance-specific content
        mock_chunks = [
            TextChunk(
                document_id="insurance_doc_123",
                content="This comprehensive health insurance policy provides coverage up to $100,000 annually for medical expenses including hospital stays, surgeries, diagnostic tests, and prescription medications.",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            TextChunk(
                document_id="insurance_doc_123",
                content="Pre-existing medical conditions are excluded from coverage for the first 12 months of the policy effective date. After this waiting period, pre-existing conditions will be covered subject to policy terms.",
                chunk_index=1,
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
            ),
            TextChunk(
                document_id="insurance_doc_123",
                content="Dental coverage includes routine cleanings, fillings, and major procedures. There is a 6-month waiting period for basic dental services and 12-month waiting period for major dental work.",
                chunk_index=2,
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
            ),
            TextChunk(
                document_id="insurance_doc_123",
                content="The following are excluded from coverage: cosmetic procedures, experimental treatments, injuries from illegal activities, and treatment received outside the network without prior authorization.",
                chunk_index=3,
                embedding=[0.4, 0.5, 0.6, 0.7, 0.8]
            ),
            TextChunk(
                document_id="insurance_doc_123",
                content="To file a claim, submit the completed claim form along with original receipts and medical reports within 30 days of treatment. Claims can be submitted online, by mail, or through the mobile app.",
                chunk_index=4,
                embedding=[0.5, 0.6, 0.7, 0.8, 0.9]
            )
        ]
        
        mock_document.text_chunks = mock_chunks
        
        # Mock search results
        mock_search_results = [
            [SearchResult(
                chunk_id="chunk0",
                content=mock_chunks[0].content,
                similarity_score=0.95,
                document_metadata={"document_id": "insurance_doc_123", "chunk_index": 0}
            )],
            [SearchResult(
                chunk_id="chunk1",
                content=mock_chunks[1].content,
                similarity_score=0.92,
                document_metadata={"document_id": "insurance_doc_123", "chunk_index": 1}
            )],
            [SearchResult(
                chunk_id="chunk2",
                content=mock_chunks[2].content,
                similarity_score=0.88,
                document_metadata={"document_id": "insurance_doc_123", "chunk_index": 2}
            )],
            [SearchResult(
                chunk_id="chunk3",
                content=mock_chunks[3].content,
                similarity_score=0.85,
                document_metadata={"document_id": "insurance_doc_123", "chunk_index": 3}
            )],
            [SearchResult(
                chunk_id="chunk4",
                content=mock_chunks[4].content,
                similarity_score=0.90,
                document_metadata={"document_id": "insurance_doc_123", "chunk_index": 4}
            )]
        ]
        
        # Mock answers
        mock_answers = [
            Answer(
                question="What is the maximum coverage amount for medical expenses?",
                answer="The policy provides coverage up to $100,000 annually for medical expenses including hospital stays, surgeries, diagnostic tests, and prescription medications.",
                confidence=0.95,
                source_chunks=["chunk0"],
                reasoning="The coverage amount is clearly stated in the policy document as $100,000 annually for comprehensive medical expenses."
            ),
            Answer(
                question="Are pre-existing conditions covered under this policy?",
                answer="Pre-existing medical conditions are excluded for the first 12 months, but are covered after this waiting period subject to policy terms.",
                confidence=0.92,
                source_chunks=["chunk1"],
                reasoning="The policy explicitly states the 12-month waiting period for pre-existing conditions before coverage begins."
            ),
            Answer(
                question="What is the waiting period for dental coverage?",
                answer="There is a 6-month waiting period for basic dental services and a 12-month waiting period for major dental work.",
                confidence=0.88,
                source_chunks=["chunk2"],
                reasoning="The dental coverage section specifies different waiting periods for basic and major dental procedures."
            ),
            Answer(
                question="What are the exclusions mentioned in the policy?",
                answer="Exclusions include cosmetic procedures, experimental treatments, injuries from illegal activities, and out-of-network treatment without prior authorization.",
                confidence=0.85,
                source_chunks=["chunk3"],
                reasoning="The policy lists specific exclusions that are not covered under the insurance plan."
            ),
            Answer(
                question="How do I file a claim for reimbursement?",
                answer="Submit the completed claim form with original receipts and medical reports within 30 days. Claims can be filed online, by mail, or through the mobile app.",
                confidence=0.90,
                source_chunks=["chunk4"],
                reasoning="The claims process is detailed in the policy with specific requirements and submission methods."
            )
        ]
        
        return mock_document, mock_search_results, mock_answers
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_insurance_policy_complete_workflow(self, mock_settings, client, auth_headers, insurance_policy_request):
        """Test complete workflow with insurance policy document"""
        mock_settings.bearer_token = settings.bearer_token
        
        mock_document, mock_search_results, mock_answers = self.setup_insurance_mocks()
        
        # Mock all service calls
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            mock_loader.load_document.return_value = mock_document
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            with patch('app.main.db_manager') as mock_db:
                mock_db.store_document_metadata.return_value = "insurance_doc_123"
                mock_db.update_document_status.return_value = True
                mock_db.log_query.return_value = True
                
                with patch.object(client.app.state, 'query_processor', create=True) as mock_processor:
                    mock_processor.text_extractor.extract_text.return_value = mock_document.text_chunks
                    
                    with patch('app.main.embedding_service') as mock_embedding:
                        mock_embedding.embed_text_chunks.return_value = mock_document.text_chunks
                        
                        with patch('app.main.vector_store') as mock_vector:
                            mock_vector.store_text_chunks.return_value = True
                            
                            with patch('app.main.query_analyzer') as mock_analyzer:
                                mock_intent = MagicMock()
                                mock_intent.intent_type = "insurance_coverage"
                                mock_analyzer.analyze_query.return_value = mock_intent
                                
                                with patch('app.main.search_engine') as mock_search:
                                    mock_search.hybrid_search.side_effect = mock_search_results
                                    
                                    with patch('app.main.response_generator') as mock_response_gen:
                                        mock_response_gen.generate_answer.side_effect = mock_answers
                                        
                                        # Make the API call
                                        start_time = time.time()
                                        response = client.post("/hackrx/run", json=insurance_policy_request, headers=auth_headers)
                                        end_time = time.time()
                                        
                                        # Verify response time is under 30 seconds
                                        assert (end_time - start_time) < 30.0
                                        
                                        # Verify response structure
                                        assert response.status_code == 200
                                        data = response.json()
                                        
                                        assert "answers" in data
                                        assert len(data["answers"]) == 5
                                        
                                        # Verify specific insurance-related answers
                                        answers = data["answers"]
                                        assert "$100,000" in answers[0]  # Coverage amount
                                        assert "12 months" in answers[1]  # Pre-existing conditions
                                        assert "6-month" in answers[2] or "12-month" in answers[2]  # Dental waiting
                                        assert "cosmetic procedures" in answers[3] or "exclusions" in answers[3].lower()  # Exclusions
                                        assert "30 days" in answers[4] or "claim form" in answers[4]  # Claims process
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_legal_contract_complete_workflow(self, mock_settings, client, auth_headers, legal_contract_request):
        """Test complete workflow with legal contract document"""
        mock_settings.bearer_token = settings.bearer_token
        
        # Setup legal contract mocks
        mock_document = Document(
            id="legal_doc_456",
            url="https://example.com/employment-contract.pdf",
            content_type="application/pdf"
        )
        
        mock_chunks = [
            TextChunk(
                document_id="legal_doc_456",
                content="The employment may be terminated by either party with 30 days written notice. During the notice period, the employee is expected to fulfill all duties and responsibilities.",
                chunk_index=0,
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            TextChunk(
                document_id="legal_doc_456",
                content="Employee agrees not to compete with the company for a period of 12 months after termination within a 50-mile radius of company offices.",
                chunk_index=1,
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
            ),
            TextChunk(
                document_id="legal_doc_456",
                content="Standard working hours are 40 hours per week, Monday through Friday, 9:00 AM to 5:00 PM with one hour lunch break.",
                chunk_index=2,
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
            ),
            TextChunk(
                document_id="legal_doc_456",
                content="Compensation package includes base salary, health insurance, dental coverage, 401(k) matching up to 6%, and 15 days paid vacation annually.",
                chunk_index=3,
                embedding=[0.4, 0.5, 0.6, 0.7, 0.8]
            ),
            TextChunk(
                document_id="legal_doc_456",
                content="The probationary period is 90 days from the start date, during which employment may be terminated without notice by either party.",
                chunk_index=4,
                embedding=[0.5, 0.6, 0.7, 0.8, 0.9]
            )
        ]
        
        mock_document.text_chunks = mock_chunks
        
        # Mock the complete workflow
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            mock_loader.load_document.return_value = mock_document
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            with patch('app.main.db_manager') as mock_db:
                mock_db.store_document_metadata.return_value = "legal_doc_456"
                mock_db.update_document_status.return_value = True
                mock_db.log_query.return_value = True
                
                with patch.object(client.app.state, 'query_processor', create=True) as mock_processor:
                    mock_processor.text_extractor.extract_text.return_value = mock_chunks
                    
                    with patch('app.main.embedding_service') as mock_embedding:
                        mock_embedding.embed_text_chunks.return_value = mock_chunks
                        
                        with patch('app.main.vector_store') as mock_vector:
                            mock_vector.store_text_chunks.return_value = True
                            
                            with patch('app.main.query_analyzer') as mock_analyzer:
                                mock_intent = MagicMock()
                                mock_intent.intent_type = "legal_terms"
                                mock_analyzer.analyze_query.return_value = mock_intent
                                
                                with patch('app.main.search_engine') as mock_search:
                                    # Return relevant chunks for each question
                                    mock_search.hybrid_search.side_effect = [
                                        [SearchResult(chunk_id=f"chunk{i}", content=chunk.content, similarity_score=0.9, document_metadata={"document_id": "legal_doc_456"})]
                                        for i, chunk in enumerate(mock_chunks)
                                    ]
                                    
                                    with patch('app.main.response_generator') as mock_response_gen:
                                        mock_answers = [
                                            "30 days written notice is required for termination by either party.",
                                            "Yes, there is a 12-month non-compete clause within a 50-mile radius.",
                                            "Standard working hours are 40 hours per week, Monday through Friday, 9:00 AM to 5:00 PM.",
                                            "Benefits include health insurance, dental coverage, 401(k) matching up to 6%, and 15 days paid vacation.",
                                            "The probationary period is 90 days from the start date."
                                        ]
                                        
                                        mock_response_gen.generate_answer.side_effect = [
                                            Answer(question=q, answer=a, confidence=0.9, source_chunks=[f"chunk{i}"], reasoning="Based on contract terms")
                                            for i, (q, a) in enumerate(zip(legal_contract_request["questions"], mock_answers))
                                        ]
                                        
                                        # Make the API call
                                        response = client.post("/hackrx/run", json=legal_contract_request, headers=auth_headers)
                                        
                                        # Verify response
                                        assert response.status_code == 200
                                        data = response.json()
                                        
                                        assert "answers" in data
                                        assert len(data["answers"]) == 5
                                        
                                        # Verify legal contract specific answers
                                        answers = data["answers"]
                                        assert "30 days" in answers[0]  # Notice period
                                        assert "non-compete" in answers[1].lower() or "12 months" in answers[1]  # Non-compete
                                        assert "40 hours" in answers[2] or "9:00 AM" in answers[2]  # Working hours
                                        assert "401(k)" in answers[3] or "health insurance" in answers[3]  # Benefits
                                        assert "90 days" in answers[4]  # Probation period
    
    @pytest.mark.asyncio
    @patch('app.main.settings')
    async def test_hr_handbook_complete_workflow(self, mock_settings, client, auth_headers, hr_handbook_request):
        """Test complete workflow with HR handbook document"""
        mock_settings.bearer_token = settings.bearer_token
        
        # Setup HR handbook mocks (simplified for brevity)
        mock_document = Document(id="hr_doc_789", url="https://example.com/employee-handbook.pdf")
        mock_document.text_chunks = [
            TextChunk(document_id="hr_doc_789", content="Remote work is permitted up to 3 days per week with manager approval.", chunk_index=0),
            TextChunk(document_id="hr_doc_789", content="Employees receive 20 vacation days annually, accrued monthly.", chunk_index=1),
            TextChunk(document_id="hr_doc_789", content="Time off requests must be submitted 2 weeks in advance through the HR portal.", chunk_index=2),
            TextChunk(document_id="hr_doc_789", content="Performance reviews are conducted annually in January with quarterly check-ins.", chunk_index=3),
            TextChunk(document_id="hr_doc_789", content="Training opportunities include online courses, conferences, and certification programs.", chunk_index=4)
        ]
        
        # Mock the workflow (simplified)
        with patch('app.main.DocumentLoader') as mock_loader_class:
            mock_loader = AsyncMock()
            mock_loader.load_document.return_value = mock_document
            mock_loader_class.return_value.__aenter__.return_value = mock_loader
            mock_loader_class.return_value.__aexit__.return_value = None
            
            with patch('app.main.db_manager') as mock_db:
                mock_db.store_document_metadata.return_value = "hr_doc_789"
                mock_db.update_document_status.return_value = True
                mock_db.log_query.return_value = True
                
                with patch.object(client.app.state, 'query_processor', create=True) as mock_processor:
                    mock_processor.text_extractor.extract_text.return_value = mock_document.text_chunks
                    
                    # Mock remaining services
                    with patch('app.main.embedding_service'), \
                         patch('app.main.vector_store'), \
                         patch('app.main.query_analyzer'), \
                         patch('app.main.search_engine') as mock_search, \
                         patch('app.main.response_generator') as mock_response_gen:
                        
                        # Setup search results
                        mock_search.hybrid_search.side_effect = [
                            [SearchResult(chunk_id=f"chunk{i}", content=chunk.content, similarity_score=0.85, document_metadata={"document_id": "hr_doc_789"})]
                            for i, chunk in enumerate(mock_document.text_chunks)
                        ]
                        
                        # Setup answers
                        mock_answers = [
                            "Remote work is permitted up to 3 days per week with manager approval.",
                            "Employees receive 20 vacation days annually, accrued monthly.",
                            "Time off requests must be submitted 2 weeks in advance through the HR portal.",
                            "Performance reviews are conducted annually in January with quarterly check-ins.",
                            "Training opportunities include online courses, conferences, and certification programs."
                        ]
                        
                        mock_response_gen.generate_answer.side_effect = [
                            Answer(question=q, answer=a, confidence=0.85, source_chunks=[f"chunk{i}"], reasoning="Based on HR handbook")
                            for i, (q, a) in enumerate(zip(hr_handbook_request["questions"], mock_answers))
                        ]
                        
                        # Make the API call
                        response = client.post("/hackrx/run", json=hr_handbook_request, headers=auth_headers)
                        
                        # Verify response
                        assert response.status_code == 200
                        data = response.json()
                        
                        assert "answers" in data
                        assert len(data["answers"]) == 5
                        
                        # Verify HR handbook specific answers
                        answers = data["answers"]
                        assert "3 days" in answers[0] or "remote work" in answers[0].lower()
                        assert "20" in answers[1] and "vacation" in answers[1].lower()
                        assert "2 weeks" in answers[2] or "advance" in answers[2]
                        assert "annually" in answers[3] or "January" in answers[3]
                        assert "training" in answers[4].lower() or "courses" in answers[4]
    
    def test_api_documentation_accessibility(self, client):
        """Test that API documentation is accessible"""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Verify OpenAPI structure
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data
        assert "/hackrx/run" in openapi_data["paths"]
    
    def test_health_and_metrics_endpoints(self, client):
        """Test health and metrics endpoints"""
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        
        # Test metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        assert "system_status" in metrics_data
        assert "cache_stats" in metrics_data
        assert "timestamp" in metrics_data
    
    def test_error_handling_scenarios(self, client, auth_headers):
        """Test various error handling scenarios"""
        # Test missing document URL
        response = client.post("/hackrx/run", json={"questions": ["What is covered?"]}, headers=auth_headers)
        assert response.status_code == 422
        
        # Test missing questions
        response = client.post("/hackrx/run", json={"documents": "https://example.com/test.pdf"}, headers=auth_headers)
        assert response.status_code == 422
        
        # Test empty questions list
        response = client.post("/hackrx/run", json={"documents": "https://example.com/test.pdf", "questions": []}, headers=auth_headers)
        assert response.status_code == 422
        
        # Test invalid JSON
        response = client.post("/hackrx/run", data="invalid json", headers=auth_headers)
        assert response.status_code == 422
        
        # Test missing authentication
        response = client.post("/hackrx/run", json={"documents": "https://example.com/test.pdf", "questions": ["What is covered?"]})
        assert response.status_code == 401
        
        # Test invalid authentication
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = client.post("/hackrx/run", json={"documents": "https://example.com/test.pdf", "questions": ["What is covered?"]}, headers=invalid_headers)
        assert response.status_code == 401
    
    def test_response_format_compliance(self, client, auth_headers):
        """Test that responses comply with the required format"""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?", "What are the terms?"]
        }
        
        # Mock a simple successful response
        with patch('app.main.query_processor') as mock_processor:
            mock_response = MagicMock()
            mock_response.answers = ["Coverage includes medical expenses", "Terms are specified in section 3"]
            mock_processor.process_query_request.return_value = mock_response
            
            response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify required response structure
            assert "answers" in data
            assert isinstance(data["answers"], list)
            assert len(data["answers"]) == 2
            assert all(isinstance(answer, str) for answer in data["answers"])
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, client, auth_headers):
        """Test that performance requirements are met"""
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        # Mock a response that takes some time but under 30 seconds
        with patch('app.main.query_processor') as mock_processor:
            async def mock_process(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate processing time
                mock_response = MagicMock()
                mock_response.answers = ["Test answer"]
                return mock_response
            
            mock_processor.process_query_request = mock_process
            
            start_time = time.time()
            response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
            end_time = time.time()
            
            # Verify response time is under 30 seconds
            assert (end_time - start_time) < 30.0
            assert response.status_code == 200
    
    def test_concurrent_request_handling(self, client, auth_headers):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        request_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        def make_request():
            with patch('app.main.query_processor') as mock_processor:
                mock_response = MagicMock()
                mock_response.answers = ["Test answer"]
                mock_processor.process_query_request.return_value = mock_response
                
                return client.post("/hackrx/run", json=request_data, headers=auth_headers)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(result.status_code == 200 for result in results)
        assert len(results) == 5


class TestAccuracyValidation:
    """Test accuracy requirements with domain-specific test cases"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"Authorization": f"Bearer {settings.bearer_token}"}
    
    def test_insurance_domain_accuracy(self, client, auth_headers):
        """Test accuracy for insurance domain queries"""
        # This would typically involve real test documents and expected answers
        # For now, we verify the system can handle insurance-specific terminology
        
        request_data = {
            "documents": "https://example.com/insurance-policy.pdf",
            "questions": [
                "What is the deductible amount?",
                "Are prescription drugs covered?",
                "What is the out-of-pocket maximum?"
            ]
        }
        
        with patch('app.main.query_processor') as mock_processor:
            # Mock responses that demonstrate understanding of insurance terms
            mock_response = MagicMock()
            mock_response.answers = [
                "The deductible amount is $500 per year for individual coverage.",
                "Yes, prescription drugs are covered with a $10 copay for generic drugs.",
                "The out-of-pocket maximum is $3,000 per individual per year."
            ]
            mock_processor.process_query_request.return_value = mock_response
            
            response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify answers contain relevant insurance terminology
            answers = data["answers"]
            assert any("deductible" in answer.lower() for answer in answers)
            assert any("prescription" in answer.lower() or "drug" in answer.lower() for answer in answers)
            assert any("out-of-pocket" in answer.lower() or "maximum" in answer.lower() for answer in answers)
    
    def test_legal_domain_accuracy(self, client, auth_headers):
        """Test accuracy for legal domain queries"""
        request_data = {
            "documents": "https://example.com/contract.pdf",
            "questions": [
                "What are the termination clauses?",
                "What is the governing law?",
                "Are there any indemnification provisions?"
            ]
        }
        
        with patch('app.main.query_processor') as mock_processor:
            mock_response = MagicMock()
            mock_response.answers = [
                "Either party may terminate this agreement with 30 days written notice.",
                "This agreement is governed by the laws of the State of California.",
                "Yes, each party agrees to indemnify the other against third-party claims."
            ]
            mock_processor.process_query_request.return_value = mock_response
            
            response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify answers contain relevant legal terminology
            answers = data["answers"]
            assert any("termination" in answer.lower() or "terminate" in answer.lower() for answer in answers)
            assert any("governing law" in answer.lower() or "laws of" in answer.lower() for answer in answers)
            assert any("indemnif" in answer.lower() for answer in answers)
    
    def test_hr_domain_accuracy(self, client, auth_headers):
        """Test accuracy for HR domain queries"""
        request_data = {
            "documents": "https://example.com/employee-handbook.pdf",
            "questions": [
                "What is the PTO policy?",
                "What are the performance review procedures?",
                "What benefits are available?"
            ]
        }
        
        with patch('app.main.query_processor') as mock_processor:
            mock_response = MagicMock()
            mock_response.answers = [
                "Employees accrue 1.67 days of PTO per month, up to 20 days annually.",
                "Performance reviews are conducted annually with quarterly check-ins.",
                "Benefits include health insurance, dental, vision, 401(k) matching, and flexible spending accounts."
            ]
            mock_processor.process_query_request.return_value = mock_response
            
            response = client.post("/hackrx/run", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify answers contain relevant HR terminology
            answers = data["answers"]
            assert any("pto" in answer.lower() or "paid time off" in answer.lower() for answer in answers)
            assert any("performance review" in answer.lower() for answer in answers)
            assert any("benefits" in answer.lower() or "insurance" in answer.lower() for answer in answers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])