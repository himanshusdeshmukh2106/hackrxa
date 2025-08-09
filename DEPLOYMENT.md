# Deployment Guide

This guide covers various deployment options for the LLM-Powered Query Retrieval System.

## Prerequisites

Before deploying, ensure you have:

1. **API Keys**:
   - Pinecone API key and environment
   - Google Gemini API key

2. **Environment Variables**:
   ```bash
   export PINECONE_API_KEY="your_pinecone_api_key"
   export PINECONE_ENVIRONMENT="us-west1-gcp"  # or your region
   export PINECONE_INDEX_NAME="document-embeddings"
   export GEMINI_API_KEY="your_gemini_api_key"
   export BEARER_TOKEN="your_secure_bearer_token"
   ```

## Deployment Options

### 1. Local Development

#### Using the Deploy Script
```bash
chmod +x deploy.sh
./deploy.sh
# Choose option 5: Setup local development
```

#### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
docker run -d \
  --name llm-postgres \
  -e POSTGRES_DB=llm_query_db \
  -e POSTGRES_USER=llm_user \
  -e POSTGRES_PASSWORD=llm_password \
  -p 5432:5432 \
  postgres:15-alpine

# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://llm_user:llm_password@localhost:5432/llm_query_db
PINECONE_API_KEY=$PINECONE_API_KEY
GEMINI_API_KEY=$GEMINI_API_KEY
BEARER_TOKEN=$BEARER_TOKEN
ENVIRONMENT=development
LOG_LEVEL=DEBUG
EOF

# Run migrations
alembic upgrade head

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Docker Deployment

#### Using Docker Compose (Recommended)
```bash
# Create .env file with your API keys
cp .env.example .env
# Edit .env with your actual values

# Build and start all services
docker-compose up --build -d

# Run database migrations
docker-compose exec app alembic upgrade head

# Check logs
docker-compose logs -f app
```

#### Using Deploy Script
```bash
./deploy.sh
# Choose option 3: Deploy with Docker (local)
```

#### Manual Docker Build
```bash
# Build the image
docker build -t llm-query-system .

# Run with environment variables
docker run -d \
  --name llm-query-app \
  -p 8000:8000 \
  -e PINECONE_API_KEY="$PINECONE_API_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e BEARER_TOKEN="$BEARER_TOKEN" \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  llm-query-system
```

### 3. Heroku Deployment

#### Using the Deploy Script
```bash
./deploy.sh
# Choose option 1: Deploy to Heroku
```

#### Manual Heroku Deployment
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-app-name

# Add PostgreSQL addon
heroku addons:create heroku-postgresql:mini

# Set environment variables
heroku config:set \
  PINECONE_API_KEY="$PINECONE_API_KEY" \
  GEMINI_API_KEY="$GEMINI_API_KEY" \
  BEARER_TOKEN="$BEARER_TOKEN" \
  ENVIRONMENT=production \
  LOG_LEVEL=INFO

# Deploy
git push heroku main

# Run migrations
heroku run alembic upgrade head

# Open the app
heroku open
```

### 4. Railway Deployment

#### Using the Deploy Script
```bash
./deploy.sh
# Choose option 2: Deploy to Railway
```

#### Manual Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Add PostgreSQL service
railway add postgresql

# Set environment variables
railway variables set \
  PINECONE_API_KEY="$PINECONE_API_KEY" \
  GEMINI_API_KEY="$GEMINI_API_KEY" \
  BEARER_TOKEN="$BEARER_TOKEN" \
  ENVIRONMENT=production

# Deploy
railway up
```

### 5. DigitalOcean App Platform

#### Using the Deploy Script
```bash
./deploy.sh
# Choose option 4: Deploy to DigitalOcean App Platform
```

#### Manual DigitalOcean Deployment
1. Create account on DigitalOcean
2. Go to App Platform
3. Connect your GitHub repository
4. Configure build settings:
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `uvicorn app.main:app --host 0.0.0.0 --port 8080`
5. Add environment variables in the dashboard
6. Add PostgreSQL database component
7. Deploy

### 6. Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (local or cloud)
- kubectl configured
- Docker image built and pushed to registry

#### Deployment Steps
```bash
# Build and push Docker image
docker build -t your-registry/llm-query-system:latest .
docker push your-registry/llm-query-system:latest

# Update kubernetes.yaml with your image and secrets
# Replace base64 encoded secrets with actual values:
echo -n "your_pinecone_api_key" | base64
echo -n "your_gemini_api_key" | base64

# Apply Kubernetes manifests
kubectl apply -f kubernetes.yaml

# Check deployment status
kubectl get pods -n llm-query-system
kubectl get services -n llm-query-system

# Run database migrations
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-$(date +%s)
  namespace: llm-query-system
spec:
  template:
    spec:
      containers:
      - name: migration
        image: your-registry/llm-query-system:latest
        command: ["alembic", "upgrade", "head"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: DATABASE_URL
      restartPolicy: OnFailure
EOF

# Access the application
kubectl port-forward service/llm-query-app 8000:80 -n llm-query-system
```

### 7. AWS ECS Deployment

#### Create ECS Task Definition
```json
{
  "family": "llm-query-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "llm-query-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/llm-query-system:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "secrets": [
        {"name": "PINECONE_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:pinecone-api-key"},
        {"name": "GEMINI_API_KEY", "valueFrom": "arn:aws:secretsmanager:region:account:secret:gemini-api-key"},
        {"name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:region:account:secret:database-url"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-query-system",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Environment Variables Reference

### Required Variables
- `PINECONE_API_KEY`: Your Pinecone API key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `DATABASE_URL`: PostgreSQL connection string

### Optional Variables
- `BEARER_TOKEN`: API authentication token (default: auto-generated)
- `PINECONE_ENVIRONMENT`: Pinecone environment (default: us-west1-gcp)
- `PINECONE_INDEX_NAME`: Pinecone index name (default: document-embeddings)
- `ENVIRONMENT`: Application environment (development/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `MAX_DOCUMENT_SIZE_MB`: Maximum document size in MB (default: 50)
- `EMBEDDING_DIMENSION`: Embedding vector dimension (default: 384)
- `SIMILARITY_THRESHOLD`: Similarity search threshold (default: 0.7)
- `MAX_SEARCH_RESULTS`: Maximum search results (default: 10)
- `MAX_CONTEXT_CHUNKS`: Maximum context chunks for LLM (default: 5)
- `MIN_CONFIDENCE_THRESHOLD`: Minimum confidence for answers (default: 0.6)
- `RATE_LIMIT_REQUESTS`: Rate limit per window (default: 100)
- `RATE_LIMIT_WINDOW_SECONDS`: Rate limit window in seconds (default: 3600)
- `REQUEST_TIMEOUT_SECONDS`: Request timeout (default: 30)
- `RETRY_ATTEMPTS`: Number of retry attempts (default: 3)
- `MAX_CHUNK_SIZE`: Maximum text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Text chunk overlap (default: 200)

## Health Checks and Monitoring

### Health Check Endpoints
- `GET /health`: Basic health check
- `GET /metrics`: System metrics and performance data

### Monitoring Setup
The application includes comprehensive monitoring:
- Request/response metrics
- Performance tracking
- Error rate monitoring
- Circuit breaker status
- Cache statistics

### Logging
Structured JSON logging in production with the following fields:
- Timestamp
- Log level
- Logger name
- Message
- Module/function information
- Additional context fields

## Security Considerations

### API Security
- Bearer token authentication required for all endpoints except health checks
- Rate limiting to prevent abuse
- CORS configuration for web clients
- Request/response logging for audit trails

### Infrastructure Security
- Non-root user in Docker containers
- Secrets management for sensitive data
- Network policies in Kubernetes
- HTTPS/TLS termination at load balancer

### Database Security
- Connection pooling with limits
- Prepared statements to prevent SQL injection
- Database user with minimal required permissions

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check database connectivity
docker-compose exec app python -c "
from app.services.database import db_manager
import asyncio
asyncio.run(db_manager.health_check())
"
```

#### 2. API Key Issues
```bash
# Verify environment variables
docker-compose exec app env | grep -E "(PINECONE|GEMINI)"
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats
kubectl top pods -n llm-query-system
```

#### 4. Performance Issues
```bash
# Check application metrics
curl http://localhost:8000/metrics
```

### Log Analysis
```bash
# Docker Compose logs
docker-compose logs -f app

# Kubernetes logs
kubectl logs -f deployment/llm-query-app -n llm-query-system

# Filter for errors
docker-compose logs app | grep ERROR
```

## Scaling Considerations

### Horizontal Scaling
- Stateless application design allows multiple instances
- Load balancer distributes requests
- Shared database and cache layers

### Vertical Scaling
- Increase CPU/memory for compute-intensive operations
- Monitor resource usage and adjust limits

### Database Scaling
- Connection pooling configured for concurrent access
- Read replicas for read-heavy workloads
- Database indexing for query performance

### Caching Strategy
- Multi-level caching (embedding, document, query)
- TTL-based expiration
- LRU eviction policy
- Cache warming strategies

## Backup and Recovery

### Database Backups
```bash
# Docker Compose backup
docker-compose exec postgres pg_dump -U llm_user llm_query_db > backup.sql

# Kubernetes backup
kubectl exec -n llm-query-system postgres-pod -- pg_dump -U llm_user llm_query_db > backup.sql
```

### Application Data
- Document processing logs
- Cache data (can be regenerated)
- Configuration settings

### Disaster Recovery
- Database restoration procedures
- Application redeployment
- Cache warming after recovery

## Performance Optimization

### Application Level
- Async processing for I/O operations
- Connection pooling for external services
- Batch processing for embeddings
- Circuit breakers for fault tolerance

### Infrastructure Level
- CDN for static assets
- Load balancing for high availability
- Auto-scaling based on metrics
- Resource limits and requests

### Database Level
- Proper indexing strategy
- Query optimization
- Connection pooling
- Read replicas for scaling

## Support and Maintenance

### Regular Maintenance
- Update dependencies regularly
- Monitor security vulnerabilities
- Review and rotate API keys
- Clean up old logs and data

### Monitoring Alerts
Set up alerts for:
- High error rates
- Response time degradation
- Resource utilization
- Service availability

### Documentation Updates
- Keep deployment docs current
- Document configuration changes
- Update troubleshooting guides
- Maintain runbooks for common operations