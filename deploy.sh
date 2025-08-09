#!/bin/bash

# LLM Query Retrieval System Deployment Script
# This script helps deploy the application to various cloud platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate environment variables
validate_env_vars() {
    local required_vars=("PINECONE_API_KEY" "GEMINI_API_KEY")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Please set these variables before deploying:"
        echo "export PINECONE_API_KEY=your_pinecone_api_key"
        echo "export GEMINI_API_KEY=your_gemini_api_key"
        exit 1
    fi
}

# Function to deploy to Heroku
deploy_heroku() {
    print_status "Deploying to Heroku..."
    
    if ! command_exists heroku; then
        print_error "Heroku CLI not found. Please install it first."
        exit 1
    fi
    
    # Check if Heroku app exists
    read -p "Enter your Heroku app name: " app_name
    
    if [[ -z "$app_name" ]]; then
        print_error "App name is required"
        exit 1
    fi
    
    # Set environment variables
    print_status "Setting environment variables..."
    heroku config:set \
        PINECONE_API_KEY="$PINECONE_API_KEY" \
        GEMINI_API_KEY="$GEMINI_API_KEY" \
        BEARER_TOKEN="${BEARER_TOKEN:-91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69}" \
        ENVIRONMENT=production \
        LOG_LEVEL=INFO \
        --app "$app_name"
    
    # Add PostgreSQL addon
    print_status "Adding PostgreSQL addon..."
    heroku addons:create heroku-postgresql:mini --app "$app_name" || true
    
    # Deploy
    print_status "Deploying application..."
    git push heroku main
    
    # Run migrations
    print_status "Running database migrations..."
    heroku run alembic upgrade head --app "$app_name"
    
    print_success "Deployment to Heroku completed!"
    heroku open --app "$app_name"
}

# Function to deploy to Railway
deploy_railway() {
    print_status "Deploying to Railway..."
    
    if ! command_exists railway; then
        print_error "Railway CLI not found. Please install it first."
        print_status "Install with: npm install -g @railway/cli"
        exit 1
    fi
    
    # Login to Railway
    railway login
    
    # Create new project or link existing
    read -p "Create new project? (y/n): " create_new
    
    if [[ "$create_new" == "y" ]]; then
        railway init
    else
        railway link
    fi
    
    # Add PostgreSQL service
    print_status "Adding PostgreSQL service..."
    railway add postgresql
    
    # Set environment variables
    print_status "Setting environment variables..."
    railway variables set \
        PINECONE_API_KEY="$PINECONE_API_KEY" \
        GEMINI_API_KEY="$GEMINI_API_KEY" \
        BEARER_TOKEN="${BEARER_TOKEN:-91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69}" \
        ENVIRONMENT=production \
        LOG_LEVEL=INFO
    
    # Deploy
    print_status "Deploying application..."
    railway up
    
    print_success "Deployment to Railway completed!"
    railway open
}

# Function to deploy with Docker
deploy_docker() {
    print_status "Deploying with Docker..."
    
    if ! command_exists docker; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Create .env file
    print_status "Creating .env file..."
    cat > .env << EOF
PINECONE_API_KEY=$PINECONE_API_KEY
GEMINI_API_KEY=$GEMINI_API_KEY
BEARER_TOKEN=${BEARER_TOKEN:-91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69}
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF
    
    # Build and run with docker-compose
    print_status "Building and starting services..."
    docker-compose up --build -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Run migrations
    print_status "Running database migrations..."
    docker-compose exec app alembic upgrade head
    
    print_success "Docker deployment completed!"
    print_status "Application is running at http://localhost:8000"
    print_status "API documentation available at http://localhost:8000/docs"
}

# Function to deploy to DigitalOcean App Platform
deploy_digitalocean() {
    print_status "Deploying to DigitalOcean App Platform..."
    
    if ! command_exists doctl; then
        print_error "DigitalOcean CLI (doctl) not found. Please install it first."
        exit 1
    fi
    
    # Create app spec
    print_status "Creating app specification..."
    cat > app-spec.yaml << EOF
name: llm-query-retrieval-system
services:
- name: api
  source_dir: /
  github:
    repo: your-username/your-repo
    branch: main
  run_command: uvicorn app.main:app --host 0.0.0.0 --port 8080
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 8080
  health_check:
    http_path: /health
  envs:
  - key: PINECONE_API_KEY
    value: $PINECONE_API_KEY
  - key: GEMINI_API_KEY
    value: $GEMINI_API_KEY
  - key: BEARER_TOKEN
    value: ${BEARER_TOKEN:-91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69}
  - key: ENVIRONMENT
    value: production
  - key: LOG_LEVEL
    value: INFO
databases:
- name: db
  engine: PG
  version: "15"
  size: basic-xs
EOF
    
    # Deploy
    print_status "Creating DigitalOcean app..."
    doctl apps create app-spec.yaml
    
    print_success "Deployment to DigitalOcean initiated!"
    print_status "Check the DigitalOcean dashboard for deployment status."
}

# Function to run local development setup
setup_local() {
    print_status "Setting up local development environment..."
    
    # Check if Python is installed
    if ! command_exists python3; then
        print_error "Python 3 not found. Please install Python 3.11 or later."
        exit 1
    fi
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    
    # Create .env file
    print_status "Creating .env file..."
    cat > .env << EOF
DATABASE_URL=postgresql://llm_user:llm_password@localhost:5432/llm_query_db
PINECONE_API_KEY=$PINECONE_API_KEY
GEMINI_API_KEY=$GEMINI_API_KEY
BEARER_TOKEN=${BEARER_TOKEN:-91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69}
ENVIRONMENT=development
LOG_LEVEL=DEBUG
EOF
    
    # Start PostgreSQL with Docker
    print_status "Starting PostgreSQL database..."
    docker run -d \
        --name llm-postgres \
        -e POSTGRES_DB=llm_query_db \
        -e POSTGRES_USER=llm_user \
        -e POSTGRES_PASSWORD=llm_password \
        -p 5432:5432 \
        postgres:15-alpine
    
    # Wait for database
    sleep 10
    
    # Run migrations
    print_status "Running database migrations..."
    alembic upgrade head
    
    print_success "Local development setup completed!"
    print_status "Start the application with: uvicorn app.main:app --reload"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if [[ ! -d "venv" ]]; then
        print_error "Virtual environment not found. Run setup first."
        exit 1
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Run tests
    pytest tests/ -v --cov=app --cov-report=html --cov-report=term
    
    print_success "Tests completed!"
    print_status "Coverage report available in htmlcov/index.html"
}

# Main menu
show_menu() {
    echo ""
    echo "LLM Query Retrieval System Deployment Script"
    echo "============================================="
    echo ""
    echo "Choose deployment option:"
    echo "1) Deploy to Heroku"
    echo "2) Deploy to Railway"
    echo "3) Deploy with Docker (local)"
    echo "4) Deploy to DigitalOcean App Platform"
    echo "5) Setup local development"
    echo "6) Run tests"
    echo "7) Exit"
    echo ""
}

# Main script
main() {
    # Check if .env file exists and source it
    if [[ -f ".env" ]]; then
        source .env
    fi
    
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                validate_env_vars
                deploy_heroku
                break
                ;;
            2)
                validate_env_vars
                deploy_railway
                break
                ;;
            3)
                validate_env_vars
                deploy_docker
                break
                ;;
            4)
                validate_env_vars
                deploy_digitalocean
                break
                ;;
            5)
                validate_env_vars
                setup_local
                break
                ;;
            6)
                run_tests
                break
                ;;
            7)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-7."
                ;;
        esac
    done
}

# Run main function
main "$@"