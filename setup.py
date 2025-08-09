#!/usr/bin/env python3
"""
Setup script for LLM Query Retrieval System
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_step(message):
    print(f"\n🔧 {message}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def check_python_version():
    """Check if Python version is 3.11+"""
    print_step("Checking Python version...")
    
    if sys.version_info < (3, 11):
        print_error(f"Python 3.11+ required, but {sys.version_info.major}.{sys.version_info.minor} found")
        return False
    
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ✓")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print_step("Creating virtual environment...")
    
    if os.path.exists("venv"):
        print_info("Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get the correct pip command for the platform"""
    if os.name == 'nt':  # Windows
        return os.path.join("venv", "Scripts", "pip")
    else:  # Unix/Linux/macOS
        return os.path.join("venv", "bin", "pip")

def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing dependencies...")
    
    pip_cmd = get_pip_command()
    
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def setup_environment_file():
    """Setup environment configuration file"""
    print_step("Setting up environment configuration...")
    
    if os.path.exists(".env"):
        print_info(".env file already exists")
        return True
    
    if os.path.exists(".env.example"):
        shutil.copy(".env.example", ".env")
        print_success(".env file created from template")
        print_info("Please edit .env file with your API keys:")
        print_info("  - PINECONE_API_KEY")
        print_info("  - GEMINI_API_KEY")
        return True
    else:
        print_error(".env.example file not found")
        return False

def create_directories():
    """Create necessary directories"""
    print_step("Creating directories...")
    
    directories = ["logs", "data", "uploads"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print_success("Directories created")
    return True

def check_docker():
    """Check if Docker is available"""
    print_step("Checking Docker availability...")
    
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print_success("Docker is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_info("Docker not found (optional for local development)")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your API keys:")
    print("   - Get Pinecone API key from: https://www.pinecone.io/")
    print("   - Get Gemini API key from: https://makersuite.google.com/app/apikey")
    
    print("\n2. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\n3. Start the application:")
    print("   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    
    print("\n4. Test the system:")
    print("   python validate_system.py")
    
    print("\n5. Access the API:")
    print("   - API: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")
    print("   - Health: http://localhost:8000/health")
    
    print("\n📁 Document Processing:")
    print("   - The system processes documents via URLs")
    print("   - For local files, use the file server (see README)")
    print("   - Supported formats: PDF, DOCX, TXT")

def main():
    """Main setup function"""
    print("🚀 LLM Query Retrieval System Setup")
    print("="*60)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        create_virtual_environment,
        install_dependencies,
        setup_environment_file,
        create_directories,
        check_docker
    ]
    
    for step in steps:
        if not step():
            print_error("Setup failed!")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()