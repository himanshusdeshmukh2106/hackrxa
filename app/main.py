"""
FastAPI main application
"""
import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.core.exceptions import (
    DocumentProcessingError, EmbeddingGenerationError, 
    VectorStoreError, LLMServiceError, ResponseGenerationError
)
from app.core.monitoring import system_monitor
from app.core.cache import cache_manager
from app.core.async_utils import AsyncRetry, timeout_after, gather_with_concurrency
from app.middleware.auth import AuthenticationMiddleware, RequestLoggingMiddleware, get_current_user
from app.middleware.error_handling import ErrorHandlingMiddleware, ResponseTimeMiddleware, HealthCheckMiddleware
from app.schemas.requests import QueryRequest
from app.schemas.responses import QueryResponse, ErrorResponse, HealthResponse
from app.services.database import db_manager
from app.services.document_loader import DocumentLoader
from app.services.text_extractor import TextExtractor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store, Vector
from app.services.query_analyzer import query_analyzer
from app.services.search_engine import search_engine
from app.services.llm_service import llm_service
from app.services.response_generator import response_generator


class QueryProcessor(LoggerMixin):
    """Main query processing orchestrator"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.max_processing_time = 30000  # 30 seconds in milliseconds
        self.retry_handler = AsyncRetry(max_attempts=3, base_delay=1.0)
    
    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        """
        Process a complete query request
        
        Args:
            request: QueryRequest with document URL and questions
            
        Returns:
            QueryResponse with answers
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query request with {len(request.questions)} questions")
            
            # Step 1 & 2: Load, process, and embed document in parallel with question processing
            document_processing_task = asyncio.create_task(self._load_and_process_document_with_embeddings(request.documents))
            
            # Step 3: Process all questions in parallel
            question_processing_task = asyncio.create_task(self._process_questions_after_document(document_processing_task, request.questions))
            
            # Wait for both tasks to complete
            document, answers = await asyncio.gather(document_processing_task, question_processing_task)
            
            # Step 4: Log the query
            processing_time = int((time.time() - start_time) * 1000)
            await self._log_query_batch(request, answers, processing_time)
            
            # Check processing time limit
            if processing_time > self.max_processing_time:
                self.logger.warning(f"Processing time {processing_time}ms exceeded limit {self.max_processing_time}ms")
            
            self.logger.info(f"Successfully processed query request in {processing_time}ms")
            
            return QueryResponse(answers=answers)
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Query processing failed after {processing_time}ms: {str(e)}")
            
            # Log failed query
            await self._log_failed_query(request, str(e), processing_time)
            
            raise
    
    async def _load_and_process_document_with_embeddings(self, document_url: str):
        """Load, process, and embed a document."""
        document = await self._load_and_process_document(document_url)
        await self._process_document_embeddings(document)
        return document
    
    async def _load_and_process_document(self, document_url: str):
        """Load and process document from URL"""
        try:
            self.logger.info(f"Loading document from: {document_url}")
            
            # Check cache first
            cached_document = await cache_manager.document_cache.get_document(document_url)
            if cached_document:
                self.logger.info("Using cached document")
                return cached_document
            
            # Load document with retry logic
            async with DocumentLoader() as loader:
                document = await self.retry_handler.execute(
                    loader.load_document, document_url
                )
            
            # Store document metadata in database
            document_id = await db_manager.store_document_metadata({
                "url": document.url,
                "content_type": document.content_type,
                "status": "processing",
                "metadata": document.metadata
            })
            document.id = document_id
            
            # Extract text chunks with fast method and reduced timeout
            text_chunks = await timeout_after(
                20.0,  # Reduced to 20 second timeout for speed
                self.text_extractor.extract_and_chunk_text_fast(document)
            )
            document.text_chunks = text_chunks
            
            # Cache the processed document
            await cache_manager.document_cache.set_document(
                document_url, document, ttl_seconds=3600
            )
            
            self.logger.info(f"Processed document into {len(text_chunks)} text chunks")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")
    
    async def _process_document_embeddings(self, document):
        """Generate embeddings and store in vector database"""
        try:
            self.logger.info(f"Generating embeddings for {len(document.text_chunks)} chunks")
            
            # Use circuit breaker for embedding service
            embedding_breaker = system_monitor.get_circuit_breaker("embedding")
            if embedding_breaker:
                embedded_chunks = await embedding_breaker.call(
                    embedding_service.embed_text_chunks, document.text_chunks
                )
            else:
                embedded_chunks = await embedding_service.embed_text_chunks(document.text_chunks)
            
            # Use fast vector storage with circuit breaker
            vector_breaker = system_monitor.get_circuit_breaker("pinecone")
            if vector_breaker:
                await vector_breaker.call(
                    vector_store.upsert_vectors_fast, [
                        Vector(
                            id=chunk.id,
                            values=chunk.embedding,
                            metadata={
                                "document_id": chunk.document_id,
                                "content": chunk.content[:500],  # Reduced for speed
                                "chunk_index": chunk.chunk_index
                            }
                        ) for chunk in embedded_chunks if chunk.embedding
                    ]
                )
            else:
                await vector_store.upsert_vectors_fast([
                    Vector(
                        id=chunk.id,
                        values=chunk.embedding,
                        metadata={
                            "document_id": chunk.document_id,
                            "content": chunk.content[:500],  # Reduced for speed
                            "chunk_index": chunk.chunk_index
                        }
                    ) for chunk in embedded_chunks if chunk.embedding
                ])
            
            # Update document status
            await db_manager.update_document_status(document.id, "completed")
            
            # Cache the fully processed document
            document_url = document.url
            await cache_manager.document_cache.set_document(
                document_url, document, ttl_seconds=3600
            )
            
            self.logger.info("Successfully stored document embeddings")
            
        except Exception as e:
            await db_manager.update_document_status(document.id, "failed")
            self.logger.error(f"Embedding processing failed: {str(e)}")
            raise EmbeddingGenerationError(f"Failed to process embeddings: {str(e)}")
    
    async def _process_questions_after_document(self, document_task: asyncio.Task, questions: List[str]) -> List[str]:
        """Process questions after the document has been processed."""
        document = await document_task
        return await self._process_questions(questions, document.id)
    
    async def _process_questions(self, questions: List[str], document_id: str) -> List[str]:
        """Process all questions and generate answers"""
        try:
            self.logger.info(f"Processing {len(questions)} questions")
            
            # Check cache for each question
            cached_answers = []
            uncached_questions = []
            question_indices = {}
            
            for i, question in enumerate(questions):
                cached_result = await cache_manager.query_cache.get_query_result(question, document_id)
                if cached_result:
                    cached_answers.append((i, cached_result))
                else:
                    if question not in uncached_questions:
                        uncached_questions.append(question)
                    question_indices[question] = question_indices.get(question, []) + [i]
            
            if not uncached_questions:
                # All answers were cached
                self.logger.info("All answers retrieved from cache")
                return [answer for _, answer in sorted(cached_answers)]
            
            # Analyze query intents for uncached questions with concurrency
            query_intent_tasks = [
                query_analyzer.analyze_query(question) 
                for question in uncached_questions
            ]
            query_intents = await gather_with_concurrency(5, *query_intent_tasks)
            
            # Perform semantic search for uncached questions with concurrency
            search_tasks = [
                search_engine.hybrid_search(
                    query=question,
                    query_intent=query_intents[i],
                    filters={"document_id": document_id},
                    top_k=10
                )
                for i, question in enumerate(uncached_questions)
            ]
            all_search_results = await gather_with_concurrency(3, *search_tasks)
            
            # Generate answers using batch processing when possible
            if len(uncached_questions) > 1 and self._can_use_shared_context(all_search_results):
                # Use shared context for batch processing
                shared_context = self._merge_search_results(all_search_results)
                
                # Use circuit breaker for LLM service
                llm_breaker = system_monitor.get_circuit_breaker("gemini")
                if llm_breaker:
                    answers_objects = await llm_breaker.call(
                        response_generator.generate_batch_answers,
                        uncached_questions, shared_context, query_intents
                    )
                else:
                    answers_objects = await response_generator.generate_batch_answers(
                        questions=uncached_questions,
                        context_results=shared_context,
                        query_intents=query_intents
                    )
            else:
                # Process individually with concurrency
                answer_tasks = [
                    response_generator.generate_answer(
                        context_results=all_search_results[i],
                        query=question,
                        query_intent=query_intents[i]
                    )
                    for i, question in enumerate(uncached_questions)
                ]
                answers_objects = await gather_with_concurrency(2, *answer_tasks)
            
            # Extract answer strings and cache them
            new_answers = []
            for i, (question, answer_obj) in enumerate(zip(uncached_questions, answers_objects)):
                answer_text = answer_obj.answer
                for original_index in question_indices[question]:
                    new_answers.append((original_index, answer_text))
                
                # Cache the result
                await cache_manager.query_cache.set_query_result(
                    question, document_id, answer_text, ttl_seconds=1800
                )
            
            # Combine cached and new answers
            all_answers = cached_answers + new_answers
            all_answers.sort(key=lambda x: x[0])  # Sort by original question index
            
            answers = [answer for _, answer in all_answers]
            
            self.logger.info(f"Generated {len(answers)} answers ({len(cached_answers)} from cache)")
            
            return answers
            
        except Exception as e:
            self.logger.error(f"Question processing failed: {str(e)}")
            raise ResponseGenerationError(f"Failed to process questions: {str(e)}")
    
    def _can_use_shared_context(self, all_search_results: List[List]) -> bool:
        """Check if search results can use shared context for batch processing"""
        if not all_search_results:
            return False
        
        # Check if there's significant overlap in search results
        all_chunk_ids = set()
        overlapping_chunks = set()
        
        for search_results in all_search_results:
            chunk_ids = {result.chunk_id for result in search_results}
            overlapping_chunks.update(all_chunk_ids.intersection(chunk_ids))
            all_chunk_ids.update(chunk_ids)
        
        # Use shared context if there's >30% overlap
        overlap_ratio = len(overlapping_chunks) / len(all_chunk_ids) if all_chunk_ids else 0
        return overlap_ratio > 0.3
    
    def _merge_search_results(self, all_search_results: List[List]) -> List:
        """Merge search results from multiple queries"""
        merged_results = {}
        
        for search_results in all_search_results:
            for result in search_results:
                if result.chunk_id not in merged_results:
                    merged_results[result.chunk_id] = result
                else:
                    # Keep the result with higher similarity score
                    if result.similarity_score > merged_results[result.chunk_id].similarity_score:
                        merged_results[result.chunk_id] = result
        
        # Sort by similarity score
        return sorted(merged_results.values(), key=lambda x: x.similarity_score, reverse=True)
    
    async def _log_query_batch(self, request: QueryRequest, answers: List[str], processing_time_ms: int):
        """Log successful query batch"""
        try:
            for i, (question, answer) in enumerate(zip(request.questions, answers)):
                await db_manager.log_query({
                    "query": question,
                    "response": answer,
                    "processing_time_ms": processing_time_ms,
                    "document_url": request.documents
                })
        except Exception as e:
            self.logger.error(f"Failed to log query batch: {str(e)}")
    
    async def _log_failed_query(self, request: QueryRequest, error: str, processing_time_ms: int):
        """Log failed query"""
        try:
            await db_manager.log_query({
                "query": f"BATCH: {len(request.questions)} questions",
                "response": f"ERROR: {error}",
                "processing_time_ms": processing_time_ms,
                "document_url": request.documents
            })
        except Exception as e:
            self.logger.error(f"Failed to log failed query: {str(e)}")


# Global query processor
query_processor = QueryProcessor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    from app.core.startup import startup_sequence, shutdown_sequence
    
    startup_success = await startup_sequence()
    if not startup_success:
        raise RuntimeError("Application startup failed")
    
    yield
    
    # Shutdown
    await shutdown_sequence()


# Create FastAPI application
app = FastAPI(
    title="LLM-Powered Query Retrieval System",
    description="Intelligent document processing and query answering system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters - last added is executed first)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(ResponseTimeMiddleware, max_response_time_seconds=30.0)
app.add_middleware(HealthCheckMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    AuthenticationMiddleware,
    rate_limit_requests=settings.rate_limit_requests,
    rate_limit_window=settings.rate_limit_window_seconds
)


# Exception handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="DocumentProcessingError",
            message=str(exc),
            details=getattr(exc, 'details', None),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(EmbeddingGenerationError)
async def embedding_generation_exception_handler(request: Request, exc: EmbeddingGenerationError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="EmbeddingGenerationError",
            message=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(VectorStoreError)
async def vector_store_exception_handler(request: Request, exc: VectorStoreError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="VectorStoreError",
            message=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(LLMServiceError)
async def llm_service_exception_handler(request: Request, exc: LLMServiceError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="LLMServiceError",
            message=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


@app.exception_handler(ResponseGenerationError)
async def response_generation_exception_handler(request: Request, exc: ResponseGenerationError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="ResponseGenerationError",
            message=str(exc),
            timestamp=datetime.utcnow()
        ).dict()
    )


# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "LLM-Powered Query Retrieval System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check database connection
        db_healthy = await db_manager.health_check()
        
        # Check vector store
        vector_healthy = await vector_store.health_check()
        
        # Check LLM service
        llm_healthy = await llm_service.health_check()
        
        # Check embedding service
        embedding_healthy = True
        try:
            await embedding_service.health_check()
        except:
            embedding_healthy = False
        
        # Get circuit breaker status
        circuit_breakers = {
            name: breaker.get_status() 
            for name, breaker in system_monitor.circuit_breakers.items()
        }
        
        # Determine overall status
        service_checks = [db_healthy, vector_healthy, llm_healthy, embedding_healthy]
        if all(service_checks):
            overall_status = "healthy"
        elif any(service_checks):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database=db_healthy,
            vector_store=vector_healthy,
            llm_service=llm_healthy,
            embedding_service=embedding_healthy,
            circuit_breakers=circuit_breakers
        )
        
    except Exception as e:
        return HealthResponse(
            status="error",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database=False,
            error=str(e)
        )


@app.get("/metrics")
async def get_metrics():
    """Get system metrics and performance data"""
    try:
        return {
            "system_status": system_monitor.get_system_status(),
            "cache_stats": await cache_manager.get_all_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Failed to retrieve metrics", "details": str(e)}
        )


@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> QueryResponse:
    """
    Main query processing endpoint
    
    Process a document and answer questions about it using LLM-powered semantic search.
    
    - **documents**: Blob URL of the document to process
    - **questions**: List of questions to ask about the document
    
    Returns a list of answers corresponding to the input questions.
    """
    try:
        # Validate request
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document URL is required"
            )
        
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="At least one question is required"
            )
        
        # Process the query
        response = await query_processor.process_query_request(request)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="LLM-Powered Query Retrieval System",
        version="1.0.0",
        description="""
        ## Overview
        
        Intelligent document processing and query answering system for insurance, legal, HR, and compliance domains.
        
        ## Features
        
        - **Document Processing**: Supports PDF, DOCX, and email formats
        - **Semantic Search**: Uses vector embeddings for intelligent content retrieval
        - **LLM Integration**: Powered by Gemini 2.5 Pro for accurate answer generation
        - **Explainable AI**: Provides reasoning and source traceability
        - **High Performance**: Sub-30-second response times
        
        ## Authentication
        
        All endpoints (except health and docs) require Bearer token authentication:
        
        ```
        Authorization: Bearer <your-token>
        ```
        
        ## Rate Limiting
        
        API requests are rate limited to prevent abuse. Default limits:
        - 100 requests per hour per client
        - Rate limit headers included in responses
        
        ## Error Handling
        
        The API returns structured error responses with appropriate HTTP status codes:
        - 400: Bad Request (validation errors)
        - 401: Unauthorized (authentication required)
        - 422: Unprocessable Entity (document processing errors)
        - 429: Too Many Requests (rate limit exceeded)
        - 500: Internal Server Error
        - 503: Service Unavailable (external service errors)
        """,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security requirement to protected endpoints
    for path in openapi_schema["paths"]:
        if path not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            for method in openapi_schema["paths"][path]:
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )