# 🚀 Quick Start Guide

## Prerequisites

- Python 3.11 or higher
- API Keys:
  - [Pinecone API Key](https://www.pinecone.io/) (for vector storage)
  - [Google Gemini API Key](https://makersuite.google.com/app/apikey) (for LLM)

## 1. Automated Setup

Run the setup script to automatically configure everything:

```bash
python setup.py
```

This will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Create .env configuration file
- ✅ Set up directories

## 2. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## 3. Configure API Keys

Edit the `.env` file with your API keys:

```bash
# Required API Keys
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Customize other settings
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=document-embeddings
```

### Getting API Keys

**Pinecone API Key:**
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up/Login
3. Create a new project
4. Copy your API key from the dashboard

**Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Create new API key
4. Copy the generated key

## 4. Start the Application

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 5. Test the System

```bash
# Basic system validation
python validate_system.py

# With authentication
python validate_system.py --token 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69
```

## 📁 Working with Documents

### Option 1: Use Online Documents

The system works with publicly accessible document URLs:

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/your-document.pdf",
    "questions": ["What is the main topic?"]
  }'
```

### Option 2: Serve Local Documents

For testing with your own documents:

1. **Create documents directory:**
   ```bash
   mkdir documents
   ```

2. **Place your documents in the directory:**
   ```bash
   # Copy your files
   cp /path/to/your/policy.pdf documents/
   cp /path/to/your/contract.docx documents/
   cp /path/to/your/handbook.txt documents/
   ```

3. **Start the file server:**
   ```bash
   python file_server.py --port 8080 --directory documents
   ```

4. **Use local URLs in your requests:**
   ```bash
   curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Authorization: Bearer 91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "http://localhost:8080/policy.pdf",
       "questions": ["What is the coverage amount?", "What are the exclusions?"]
     }'
   ```

## 📋 Example Usage

### Insurance Policy Query
```json
{
  "documents": "http://localhost:8080/insurance-policy.pdf",
  "questions": [
    "What is the maximum coverage amount?",
    "Are pre-existing conditions covered?",
    "What is the claims process?"
  ]
}
```

### Legal Contract Query
```json
{
  "documents": "http://localhost:8080/employment-contract.docx",
  "questions": [
    "What is the termination notice period?",
    "Are there any non-compete clauses?",
    "What benefits are included?"
  ]
}
```

### HR Handbook Query
```json
{
  "documents": "http://localhost:8080/employee-handbook.pdf",
  "questions": [
    "What is the remote work policy?",
    "How many vacation days are provided?",
    "What is the performance review process?"
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **API Key Errors**
   - Verify your API keys in `.env` file
   - Check API key permissions and quotas

3. **Document Access Errors**
   - Ensure documents are accessible via HTTP
   - Check file server is running for local documents

4. **Port Already in Use**
   ```bash
   # Use different port
   python -m uvicorn app.main:app --reload --port 8001
   ```

### Getting Help

1. Check application logs
2. Verify health status: http://localhost:8000/health
3. Review configuration in `.env` file
4. Run system validation: `python validate_system.py`

## 🎯 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Test with your documents**: Use the file server approach
3. **Monitor performance**: Check http://localhost:8000/metrics
4. **Deploy to production**: See [DEPLOYMENT.md](DEPLOYMENT.md)

## 📚 Additional Resources

- **API Documentation**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Full README**: [README.md](README.md)