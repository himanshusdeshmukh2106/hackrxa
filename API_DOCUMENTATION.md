# LLM-Powered Query Retrieval System API Documentation

## Overview

The LLM-Powered Query Retrieval System is an intelligent document processing and query answering system designed for insurance, legal, HR, and compliance domains. It processes documents and answers questions using advanced semantic search and large language models.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Authentication

All API endpoints (except health checks and documentation) require Bearer token authentication.

### Authentication Header
```
Authorization: Bearer <your-token>
```

### Getting a Token
Contact your system administrator to obtain a Bearer token for API access.

## API Endpoints

### 1. Main Query Processing Endpoint

#### `POST /hackrx/run`

Process a document and answer questions about it using LLM-powered semantic search.

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the coverage amount?",
    "What are the exclusions?",
    "What is the waiting period?"
  ]
}
```

**Request Schema:**
- `documents` (string, required): Blob URL of the document to process
- `questions` (array of strings, required): List of questions to ask about the document

**Response:**
```json
{
  "answers": [
    "The coverage amount is up to $50,000 for medical expenses.",
    "Exclusions include pre-existing conditions for the first 12 months.",
    "There is a 30-day waiting period for non-emergency treatments."
  ]
}
```

**Response Schema:**
- `answers` (array of strings): List of answers corresponding to the input questions

**Status Codes:**
- `200 OK`: Successful processing
- `400 Bad Request`: Invalid request format
- `401 Unauthorized`: Missing or invalid authentication
- `422 Unprocessable Entity`: Document processing error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: External service unavailable
- `504 Gateway Timeout`: Request timeout

**Example Request:**
```bash
curl -X POST "https://your-domain.com/hackrx/run" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/insurance-policy.pdf",
    "questions": [
      "What is the maximum coverage amount?",
      "Are pre-existing conditions covered?",
      "What is the claims process?"
    ]
  }'
```

**Example Response:**
```json
{
  "answers": [
    "The maximum coverage amount is $100,000 annually for comprehensive medical expenses including hospital stays, surgeries, and prescription medications.",
    "Pre-existing conditions are excluded for the first 12 months of coverage, after which they are covered subject to policy terms and conditions.",
    "To file a claim, submit the completed claim form with original receipts and medical reports within 30 days of treatment through our online portal, mobile app, or by mail."
  ]
}
```

### 2. Health Check Endpoint

#### `GET /health`

Check the health status of the system and its components.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-08T12:00:00Z",
  "version": "1.0.0",
  "database": true,
  "vector_store": true,
  "llm_service": true,
  "embedding_service": true,
  "circuit_breakers": {
    "pinecone": {
      "state": "closed",
      "failure_count": 0
    },
    "gemini": {
      "state": "closed",
      "failure_count": 0
    }
  }
}
```

**Status Values:**
- `healthy`: All services operational
- `degraded`: Some services experiencing issues
- `unhealthy`: Critical services down
- `error`: System error occurred

### 3. System Metrics Endpoint

#### `GET /metrics`

Get system performance metrics and statistics.

**Response:**
```json
{
  "system_status": {
    "health": {
      "status": "healthy",
      "healthy_services": 4,
      "total_services": 4
    },
    "metrics": {
      "counters": {
        "http_requests_total": 1250,
        "http_errors_total": 15
      },
      "histograms": {
        "http_request_duration_ms": {
          "count": 1250,
          "min": 45,
          "max": 28500,
          "avg": 2340,
          "p50": 1800,
          "p95": 8500,
          "p99": 15000
        }
      }
    },
    "circuit_breakers": {
      "pinecone": {
        "state": "closed",
        "failure_count": 0
      }
    }
  },
  "cache_stats": {
    "embedding_cache": {
      "total_entries": 1500,
      "max_size": 10000,
      "total_accesses": 3200
    },
    "document_cache": {
      "total_entries": 45,
      "max_size": 100,
      "total_accesses": 180
    }
  },
  "timestamp": "2025-01-08T12:00:00Z"
}
```

### 4. API Documentation Endpoints

#### `GET /docs`
Interactive API documentation (Swagger UI)

#### `GET /redoc`
Alternative API documentation (ReDoc)

#### `GET /openapi.json`
OpenAPI specification in JSON format

## Request/Response Examples

### Insurance Policy Query

**Request:**
```json
{
  "documents": "https://storage.example.com/policies/health-insurance-2024.pdf",
  "questions": [
    "What is the annual deductible?",
    "Are mental health services covered?",
    "What is the out-of-network coverage percentage?",
    "How do I appeal a claim denial?",
    "What preventive care is covered at 100%?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The annual deductible is $1,500 for individual coverage and $3,000 for family coverage.",
    "Yes, mental health services are covered with the same benefits as medical services, including therapy and counseling sessions.",
    "Out-of-network coverage is provided at 60% of allowed charges after meeting the out-of-network deductible of $3,000.",
    "To appeal a claim denial, submit a written appeal within 60 days to the address provided in your denial letter, including supporting documentation.",
    "Preventive care covered at 100% includes annual physical exams, immunizations, mammograms, colonoscopies, and routine lab work."
  ]
}
```

### Legal Contract Query

**Request:**
```json
{
  "documents": "https://storage.example.com/contracts/employment-agreement-2024.pdf",
  "questions": [
    "What is the termination notice period?",
    "Are there any non-disclosure agreements?",
    "What is the compensation structure?",
    "Are there any stock option provisions?",
    "What is the dispute resolution process?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Either party may terminate this agreement with 30 days written notice, except during the probationary period where termination may be immediate.",
    "Yes, the employee agrees to maintain confidentiality of proprietary information during employment and for 2 years after termination.",
    "Compensation includes base salary of $85,000 annually, paid bi-weekly, plus performance bonuses up to 15% of base salary.",
    "Employee is eligible for stock options after 12 months of employment, vesting over 4 years with a 1-year cliff.",
    "Disputes must first be addressed through internal mediation, followed by binding arbitration if unresolved within 30 days."
  ]
}
```

### HR Handbook Query

**Request:**
```json
{
  "documents": "https://storage.example.com/hr/employee-handbook-2024.pdf",
  "questions": [
    "What is the remote work policy?",
    "How many sick days are provided?",
    "What is the dress code policy?",
    "What training programs are available?",
    "What is the parental leave policy?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Remote work is permitted up to 3 days per week with manager approval and must maintain core collaboration hours of 10 AM to 3 PM EST.",
    "Employees receive 10 sick days annually, which can be used for personal illness or caring for immediate family members.",
    "The dress code is business casual for office days and professional attire for client meetings, with casual Fridays permitted.",
    "Training programs include online learning platforms, conference attendance budget of $2,000 annually, and internal mentorship programs.",
    "Parental leave includes 12 weeks paid leave for primary caregivers and 6 weeks for secondary caregivers, with job protection guaranteed."
  ]
}
```

## Error Handling

### Error Response Format

All error responses follow a consistent format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error description",
  "details": {
    "field": "specific_field",
    "issue": "detailed_issue_description"
  },
  "timestamp": "2025-01-08T12:00:00Z"
}
```

### Common Error Types

#### Authentication Errors (401)
```json
{
  "error": "AuthenticationError",
  "message": "Invalid authentication token",
  "timestamp": "2025-01-08T12:00:00Z"
}
```

#### Validation Errors (422)
```json
{
  "error": "ValidationError",
  "message": "Invalid request format",
  "details": {
    "field": "documents",
    "issue": "Document URL is required"
  },
  "timestamp": "2025-01-08T12:00:00Z"
}
```

#### Rate Limit Errors (429)
```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Maximum 100 requests per hour.",
  "retry_after": 1800,
  "timestamp": "2025-01-08T12:00:00Z"
}
```

#### Document Processing Errors (422)
```json
{
  "error": "DocumentProcessingError",
  "message": "Failed to process document",
  "details": {
    "url": "https://example.com/document.pdf",
    "issue": "Document format not supported"
  },
  "timestamp": "2025-01-08T12:00:00Z"
}
```

#### Service Unavailable Errors (503)
```json
{
  "error": "VectorStoreError",
  "message": "Vector store service temporarily unavailable",
  "timestamp": "2025-01-08T12:00:00Z"
}
```

#### Timeout Errors (504)
```json
{
  "error": "RequestTimeout",
  "message": "Request processing timed out",
  "timestamp": "2025-01-08T12:00:00Z"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit**: 100 requests per hour per client
- **Rate Limit Headers**: Included in all responses
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Time when rate limit resets (Unix timestamp)

### Rate Limit Headers Example
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1704715200
```

## Performance Characteristics

### Response Times
- **Target**: Sub-30-second response times for all queries
- **Typical**: 2-8 seconds for standard documents
- **Factors affecting performance**:
  - Document size and complexity
  - Number of questions
  - System load

### Document Limitations
- **Maximum file size**: 50 MB
- **Supported formats**: PDF, DOCX, TXT
- **Maximum questions per request**: 10

### Caching
The system implements intelligent caching to improve performance:
- **Document cache**: Processed documents cached for 1 hour
- **Embedding cache**: Text embeddings cached for 24 hours
- **Query cache**: Query results cached for 30 minutes

## Security

### Authentication
- Bearer token authentication required
- Tokens should be kept secure and not shared
- Tokens may have expiration dates

### Data Privacy
- Documents are processed in memory and not permanently stored
- Query logs may be retained for system improvement
- No personal data is stored beyond processing requirements

### HTTPS
- All production endpoints use HTTPS encryption
- API keys and tokens are transmitted securely

## SDKs and Client Libraries

### Python Client Example
```python
import requests

class LLMQueryClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def query_document(self, document_url, questions):
        response = requests.post(
            f"{self.base_url}/hackrx/run",
            json={
                "documents": document_url,
                "questions": questions
            },
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = LLMQueryClient("https://your-domain.com", "your-token")
result = client.query_document(
    "https://example.com/policy.pdf",
    ["What is covered?", "What are the exclusions?"]
)
print(result["answers"])
```

### JavaScript Client Example
```javascript
class LLMQueryClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async queryDocument(documentUrl, questions) {
        const response = await fetch(`${this.baseUrl}/hackrx/run`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                documents: documentUrl,
                questions: questions
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage
const client = new LLMQueryClient('https://your-domain.com', 'your-token');
client.queryDocument(
    'https://example.com/policy.pdf',
    ['What is covered?', 'What are the exclusions?']
).then(result => {
    console.log(result.answers);
});
```

### cURL Examples

#### Basic Query
```bash
curl -X POST "https://your-domain.com/hackrx/run" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?"]
  }'
```

#### Health Check
```bash
curl -X GET "https://your-domain.com/health"
```

#### Get Metrics
```bash
curl -X GET "https://your-domain.com/metrics" \
  -H "Authorization: Bearer your-token"
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures
- **Issue**: 401 Unauthorized responses
- **Solution**: Verify Bearer token is correct and not expired
- **Check**: Ensure "Bearer " prefix is included in Authorization header

#### 2. Document Processing Failures
- **Issue**: 422 Unprocessable Entity for document processing
- **Solutions**:
  - Verify document URL is accessible
  - Check document format is supported (PDF, DOCX, TXT)
  - Ensure document size is under 50 MB

#### 3. Slow Response Times
- **Issue**: Requests taking longer than expected
- **Solutions**:
  - Reduce number of questions per request
  - Use smaller documents when possible
  - Check system status at `/health` endpoint

#### 4. Rate Limiting
- **Issue**: 429 Too Many Requests
- **Solution**: Implement exponential backoff and respect rate limit headers

#### 5. Service Unavailable
- **Issue**: 503 Service Unavailable responses
- **Solution**: Check system health and retry after a delay

### Getting Help

For additional support:
1. Check the system health endpoint: `/health`
2. Review error messages and details in responses
3. Contact your system administrator
4. Check system metrics at `/metrics` endpoint

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Support for PDF, DOCX, and TXT documents
- Insurance, legal, and HR domain optimization
- Bearer token authentication
- Rate limiting and caching
- Comprehensive error handling
- Health monitoring and metrics

### Planned Features
- Support for additional document formats
- Batch processing capabilities
- Webhook notifications
- Advanced analytics and reporting
- Multi-language support