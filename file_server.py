#!/usr/bin/env python3
"""
Simple HTTP file server for serving local documents
Use this to serve your training documents locally for testing
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import argparse

class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    parser = argparse.ArgumentParser(description='Serve local documents for LLM Query System')
    parser.add_argument('--port', type=int, default=8080, help='Port to serve on (default: 8080)')
    parser.add_argument('--directory', default='documents', help='Directory to serve (default: documents)')
    
    args = parser.parse_args()
    
    # Create documents directory if it doesn't exist
    doc_dir = Path(args.directory)
    doc_dir.mkdir(exist_ok=True)
    
    # Change to the documents directory
    os.chdir(doc_dir)
    
    print(f"🗂️  Document Server Starting...")
    print(f"📁 Serving directory: {doc_dir.absolute()}")
    print(f"🌐 Server URL: http://localhost:{args.port}")
    print(f"📄 Place your documents in: {doc_dir.absolute()}")
    print("\n📋 Usage Examples:")
    print(f"   - PDF: http://localhost:{args.port}/policy.pdf")
    print(f"   - DOCX: http://localhost:{args.port}/contract.docx")
    print(f"   - TXT: http://localhost:{args.port}/handbook.txt")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Start the server
    try:
        server = HTTPServer(('localhost', args.port), CORSRequestHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()