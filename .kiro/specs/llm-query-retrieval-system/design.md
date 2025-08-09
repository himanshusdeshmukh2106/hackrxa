# Design Document

## Overview

The LLM-Powered Intelligent Query-Retrieval System is designed as a microservices architecture using FastAPI, integrating Pinecone for vector storage, Gemini 2.5 Pro for natural language processing, and PostgreSQL for persistent data management. The system follows a pipeline architecture where documents are processed, embedded, indexed, and then queried through semantic search with explainable AI capabilities.

The core workflow involves: Document ingestion → Text extraction → Embedding generation → Vector storage → Query processing → Semantic retrieval → LLM reasoning → Structured response generation.

## Architecture

### System Architecture Diagram

```mermaid
graph TB
    A[Client Request] --> B[FastAPI Gateway]
    B --> C[Authentication Middleware]
    C --> D[Document Processor]
    D --> E[Text Extractor]
    E --> F[Embedding Generator]
    F --> G[Pinecone Vector DB]
    
    B --> H[Query Processor]
    H --> I[Semantic Search Engine]
    I --> G
    I --> J[Clause Matcher]
    J --> K[Gemini 2.5 Pro LLM]
    K --> L[Response Generator]
    L --> M[JSON Formatter]
    M --> N[Client Response]
    
    O[PostgreSQL] --> P[Document Metadata]
    O --> Q[Query Logs]
    O --> R[System Config]
    
    D --> O
    H --> O
```

### Component Architecture

The system is organized into six main layers:

1. **API Layer**: FastAPI-based REST endpoints with authentication
2. **Processing Layer**: Document parsing and query processing services
3. **Intelligence Layer**: Embedding generation and LLM reasoning
4. **Storage Layer**: Pinecone for vectors, PostgreSQL for metadata
5. **Search Layer**: Semantic search and clause matching
6. **Response Layer**: JSON formatting and explainability

### Deployment Architecture

The system will be deployed as a containerized application with the following components:
- FastAPI application server (main container)
- PostgreSQL database (managed service or container)
- Pinecone cloud service (external API)
- Gemini 2.5 Pro API (external service)

### Environment Configuration

All sensitive credentials and configuration will be managed through environment variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/llm_query_db
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=llm_query_db

# API Keys
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=document_embeddings
GEMINI_API_KEY=your_gemini_api_key

# Authentication
BEARER_TOKEN=91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69
JWT_SECRET_KEY=your_jwt_secret

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_DOCUMENT_SIZE_MB=50
EMBEDDING_DIMENSION=768
```

## Components and Interfaces

### 1. API Gateway Component

**FastAPI Application (`app/main.py`)**
- Handles HTTP requests and responses
- Implements Bearer token authentication using environment variables
- Provides OpenAPI documentation
- Manages CORS and security headers
- Loads configuration from environment variables securely

**Key Interfaces:**
```python
@app.post("/hackrx/run")
async def process_query(
    request: QueryRequest,
    authorization: str = Header(...)
) -> QueryResponse
```

**Models:**
```python
class QueryRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
```

### 2. Document Processing Component

**Document Loader (`app/services/document_loader.py`)**
- Downloads documents from blob URLs
- Handles PDF, DOCX, and email formats
- Implements retry logic and error handling

**Text Extractor (`app/services/text_extractor.py`)**
- Extracts text from different document formats
- Preserves document structure and metadata
- Handles tables, headers, and formatting

**Key Interfaces:**
```python
class DocumentProcessor:
    async def load_document(self, url: str) -> Document
    async def extract_text(self, document: Document) -> List[TextChunk]
    async def preprocess_chunks(self, chunks: List[TextChunk]) -> List[ProcessedChunk]
```

### 3. Embedding and Vector Storage Component

**Embedding Generator (`app/services/embedding_service.py`)**
- Generates embeddings using sentence transformers
- Implements batch processing for efficiency
- Handles embedding dimension consistency

**Pinecone Manager (`app/services/vector_store.py`)**
- Manages Pinecone index operations
- Implements upsert and query operations
- Handles vector metadata storage

**Key Interfaces:**
```python
class EmbeddingService:
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]
    async def store_embeddings(self, embeddings: List[Embedding]) -> bool

class VectorStore:
    async def upsert_vectors(self, vectors: List[Vector]) -> bool
    async def similarity_search(self, query_vector: List[float], top_k: int) -> List[Match]
```

### 4. Query Processing Component

**Query Analyzer (`app/services/query_analyzer.py`)**
- Parses natural language queries
- Extracts intent and entities
- Generates search parameters

**Semantic Search Engine (`app/services/search_engine.py`)**
- Performs vector similarity search
- Implements hybrid search (semantic + keyword)
- Ranks and filters results

**Key Interfaces:**
```python
class QueryAnalyzer:
    async def analyze_query(self, query: str) -> QueryIntent
    async def extract_entities(self, query: str) -> List[Entity]

class SearchEngine:
    async def semantic_search(self, query: str, filters: Dict) -> List[SearchResult]
    async def rank_results(self, results: List[SearchResult]) -> List[RankedResult]
```

### 5. LLM Integration Component

**Gemini Service (`app/services/llm_service.py`)**
- Integrates with Gemini 2.5 Pro API
- Implements prompt engineering
- Handles token optimization and batching

**Response Generator (`app/services/response_generator.py`)**
- Generates contextual answers
- Provides explainability and reasoning
- Formats responses according to requirements

**Key Interfaces:**
```python
class LLMService:
    async def generate_response(self, context: str, query: str) -> LLMResponse
    async def batch_process(self, requests: List[LLMRequest]) -> List[LLMResponse]

class ResponseGenerator:
    async def generate_answer(self, context: List[SearchResult], query: str) -> Answer
    async def explain_reasoning(self, answer: Answer) -> Explanation
```

### 6. Data Management Component

**PostgreSQL Manager (`app/services/database.py`)**
- Manages database connections using environment variables
- Stores document metadata and query logs
- Handles configuration management
- Implements connection pooling and security

**Configuration Manager (`app/core/config.py`)**
- Loads environment variables securely
- Validates configuration settings
- Provides centralized settings management

**Models (`app/models/`)**
- SQLAlchemy models for data persistence
- Document, Query, and Configuration entities

**Key Interfaces:**
```python
class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
    
    async def store_document_metadata(self, metadata: DocumentMetadata) -> int
    async def log_query(self, query_log: QueryLog) -> bool
    async def get_configuration(self, key: str) -> Optional[str]

class Settings(BaseSettings):
    database_url: str
    pinecone_api_key: str
    gemini_api_key: str
    bearer_token: str
    
    class Config:
        env_file = ".env"
```

## Data Models

### Core Data Models

```python
# Document Models
class Document(BaseModel):
    id: str
    url: str
    content_type: str
    text_chunks: List[TextChunk]
    metadata: Dict[str, Any]
    created_at: datetime

class TextChunk(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]

# Query Models
class QueryIntent(BaseModel):
    original_query: str
    intent_type: str
    entities: List[Entity]
    confidence: float

class SearchResult(BaseModel):
    chunk_id: str
    content: str
    similarity_score: float
    document_metadata: Dict[str, Any]
    relevance_explanation: str

# Response Models
class Answer(BaseModel):
    question: str
    answer: str
    confidence: float
    source_chunks: List[str]
    reasoning: str
```

### Database Schema

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    content_type VARCHAR(50),
    status VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Query logs table
CREATE TABLE query_logs (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    query TEXT NOT NULL,
    response TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- System configuration table
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Error Handling

### Error Categories and Responses

1. **Authentication Errors (401)**
   - Invalid or missing Bearer token
   - Expired authentication credentials

2. **Validation Errors (422)**
   - Invalid request format
   - Missing required fields
   - Invalid document URLs

3. **Processing Errors (500)**
   - Document download failures
   - LLM API failures
   - Vector database connectivity issues

4. **Timeout Errors (504)**
   - Processing time exceeds 30-second limit
   - External service timeouts

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
```

### Retry and Fallback Strategies

- **Document Processing**: 3 retries with exponential backoff
- **LLM API Calls**: Circuit breaker pattern with fallback to cached responses
- **Vector Search**: Fallback to PostgreSQL full-text search if Pinecone unavailable
- **Embedding Generation**: Batch processing with partial failure handling

## Testing Strategy

### Unit Testing

- **Component Testing**: Each service component tested in isolation
- **Mock Dependencies**: External APIs mocked for consistent testing
- **Coverage Target**: Minimum 85% code coverage
- **Test Framework**: pytest with async support

### Integration Testing

- **API Testing**: End-to-end API endpoint testing
- **Database Testing**: PostgreSQL integration with test containers
- **Vector Store Testing**: Pinecone integration testing with test indexes
- **LLM Testing**: Gemini API integration with test prompts

### Performance Testing

- **Load Testing**: Concurrent request handling up to 100 requests/minute
- **Response Time Testing**: Sub-30-second response time validation
- **Memory Testing**: Memory usage profiling under load
- **Token Usage Testing**: LLM token consumption optimization validation

### Accuracy Testing

- **Domain-Specific Testing**: Insurance, legal, HR, and compliance document sets
- **Ground Truth Validation**: Human-annotated test cases for accuracy measurement
- **Confidence Scoring**: Validation of confidence thresholds
- **Explainability Testing**: Reasoning quality assessment

### Test Data Management

```python
# Test fixtures for different document types
@pytest.fixture
def sample_insurance_policy():
    return load_test_document("insurance_policy.pdf")

@pytest.fixture
def sample_legal_contract():
    return load_test_document("legal_contract.docx")

# Performance test scenarios
@pytest.mark.performance
async def test_concurrent_queries():
    # Test 50 concurrent queries
    pass

@pytest.mark.accuracy
async def test_insurance_query_accuracy():
    # Test accuracy on insurance domain
    pass
```

This design provides a robust, scalable, and maintainable architecture that meets all the specified requirements while leveraging the recommended tech stack effectively.