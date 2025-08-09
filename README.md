# LLM Query Retrieval System

An intelligent document query system using LLM and semantic search for insurance, legal, HR, and compliance domains.

## Features

- **Multi-format Document Processing**: Supports PDF, DOCX, and email documents
- **Semantic Search**: Uses embeddings and vector similarity for intelligent document retrieval
- **LLM Integration**: Powered by Gemini 2.5 Pro for natural language understanding
- **Explainable AI**: Provides reasoning and source traceability for all answers
- **High Performance**: Optimized for sub-30-second response times
- **Scalable Architecture**: Built with FastAPI and async processing
- **Comprehensive Monitoring**: Health checks, metrics, and structured logging

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Text           │    │   Embedding     │
│   Loader        │───▶│   Extractor      │───▶│   Generator     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Response      │    │   LLM Service    │    │   Vector Store  │
│   Generator     │◀───│   (Gemini)       │    │   (Pinecone)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        ▲                       │
         ▼                        │                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Query          │    │   Search        │
│   Endpoint      │───▶│   Analyzer       │◀───│   Engine        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- API Keys:
  - Pinecone API key
  - Google Gemini API key

### 1. Clone and Setup

```bash
git clone <repository-url>
cd llm-query-retrieval-system

# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

### 2. Set Required Environment Variables

Edit `.env` file and set:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
POSTGRES_PASSWORD=your_secure_password
```

### 3. Deploy with Docker

```bash
# Make deploy script executable (Linux/Mac)
chmod +x deploy.sh

# Deploy the application
./deploy.sh

# Or manually with docker-compose
docker-compose up --build -d
```

### 4. Verify Deployment

```bash
# Check service status
docker-compose ps

# Check application health
curl http://localhost:8000/health

# View detailed health information
curl http://localhost:8000/health/detailed
```

## API Usage

### Authentication

All API requests require a Bearer token:

```bash
Authorization: Bearer 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69
```

### Main Endpoint

**POST /hackrx/run**

Process a document and answer questions about it.

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What are the exclusions in this policy?"
    ]
  }'
```

**Response:**

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "Exclusions include pre-existing conditions and cosmetic procedures."
  ]
}
```

### Other Endpoints

- **GET /**: Root endpoint with system information
- **GET /health**: Basic health check
- **GET /health/detailed**: Comprehensive health check
- **GET /metrics**: System performance metrics
- **GET /docs**: Interactive API documentation

## Deployment

### Deploying to Render

This application is optimized for deployment on [Render](https://render.com/).

1.  **Fork this repository** to your own GitHub account.

2.  **Create a new "Web Service"** on the Render dashboard and connect it to your forked repository.

3.  **Configure the service:**
    *   **Name:** Give your service a name (e.g., `llm-query-system`).
    *   **Region:** Choose a region close to you.
    *   **Branch:** `main`
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT app.main:app`
    *   **Instance Type:** Choose a suitable instance type (the "Starter" plan is a good starting point).

4.  **Add Environment Variables:**
    *   Go to the "Environment" tab for your new service.
    *   Add the environment variables defined in `.env.example`. You will need to provide your own values for `DATABASE_URL` (if you create a Render Postgres database), `PINECONE_API_KEY`, and `GEMINI_API_KEY`.

5.  **Create a Database (Optional but Recommended):**
    *   Create a new "PostgreSQL" database on Render.
    *   Copy the "Internal Connection String" and use it for the `DATABASE_URL` environment variable in your web service.

6.  **Deploy:**
    *   Click "Create Web Service". Render will automatically build and deploy your application.

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_unit/ -v                    # Unit tests
pytest tests/test_integration.py -v          # Integration tests
pytest tests/test_performance.py -v          # Performance tests
pytest tests/test_final_integration.py -v    # End-to-end tests

# Run tests with detailed output
pytest -v --tb=short
```

### System Validation

Validate the complete system deployment:

```bash
# Basic validation (no authentication required)
python validate_system.py

# Full validation with authentication
python validate_system.py --token your_bearer_token

# Validate production system
python validate_system.py --url https://your-domain.com --token your_token

# Use configuration file
python validate_system.py --config validation_config.json
```

Example `validation_config.json`:
```json
{
  "url": "https://your-domain.com",
  "token": "your_bearer_token_here"
}
```

The validation script tests:
- Basic connectivity and health checks
- Authentication and authorization
- Request validation and error handling
- Performance characteristics
- Rate limiting functionality
- Complete system integration

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PINECONE_API_KEY` | Pinecone API key | - | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | - | Yes |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` | No |
| `BEARER_TOKEN` | API authentication token | `91d7c7f...` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `MAX_DOCUMENT_SIZE_MB` | Maximum document size | `50` | No |
| `SIMILARITY_THRESHOLD` | Vector similarity threshold | `0.7` | No |

### Performance Tuning

- `MAX_CONCURRENT_TASKS`: Maximum concurrent async tasks (default: 10)
- `MAX_THREAD_POOL_WORKERS`: Thread pool size (default: 4)
- `MAX_MEMORY_USAGE_MB`: Memory usage limit (default: 1024)
- `RATE_LIMIT_REQUESTS`: Rate limit per window (default: 100)
- `CACHE_TTL_SECONDS`: Cache time-to-live (default: 3600)

## Monitoring

### Health Checks

- **Basic**: `GET /health` - Simple up/down status
- **Detailed**: `GET /health/detailed` - Comprehensive service health
- **Metrics**: `GET /metrics` - System performance metrics

### Logging

Structured JSON logging in production:

```json
{
  "timestamp": "2025-01-08T12:00:00Z",
  "level": "INFO",
  "logger": "QueryProcessor",
  "message": "Processing query request",
  "module": "main",
  "function": "process_query",
  "extra_fields": {
    "request_id": "req-123",
    "processing_time_ms": 1500
  }
}
```

### Docker Logs

```bash
# View application logs
docker-compose logs -f app

# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres
```

## Deployment Options

### Docker Compose (Recommended)

Complete stack with PostgreSQL, Redis, and Nginx:

```bash
docker-compose up -d
```

### Heroku

```bash
# Install Heroku CLI and login
heroku create your-app-name

# Set environment variables
heroku config:set PINECONE_API_KEY=your_key
heroku config:set GEMINI_API_KEY=your_key

# Deploy
git push heroku main
```

### Railway

```bash
# Install Railway CLI
railway login
railway init
railway up
```

### AWS/GCP/Azure

Use the provided Dockerfile with your preferred container orchestration platform.

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your Pinecone and Gemini API keys are correct
   - Check API key permissions and quotas

2. **Database Connection Issues**
   - Ensure PostgreSQL is running
   - Check database credentials in environment variables

3. **Memory Issues**
   - Adjust `MAX_MEMORY_USAGE_MB` setting
   - Monitor system resources with `/metrics` endpoint

4. **Slow Response Times**
   - Check vector database performance
   - Adjust `MAX_CONCURRENT_TASKS` setting
   - Monitor with detailed health checks

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

### Support

For issues and questions:
1. Check the logs: `docker-compose logs -f app`
2. Verify health status: `curl http://localhost:8000/health/detailed`
3. Review configuration in `.env` file

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Changelog

### v1.0.0
- Initial release
- Multi-format document processing
- Semantic search with Pinecone
- Gemini 2.5 Pro integration
- FastAPI REST API
- Docker deployment
- Comprehensive monitoring