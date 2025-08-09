#!/usr/bin/env python3
"""
Debug script to trace document ID through the entire pipeline
"""
import requests
import json
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.document_loader import DocumentLoader
from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service

# Document URL
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Bearer token
BEARER_TOKEN = "91d7c7fcc021f2f76b4d43c446b643214c2c13990085887798a744e2ca692e69"

async def trace_document_processing():
    """Trace document ID through the processing pipeline"""
    print("🔍 DOCUMENT ID FLOW DEBUG")
    print("=" * 60)
    
    try:
        # Step 1: Load document
        print("📄 Step 1: Loading document...")
        async with DocumentLoader() as document_loader:
            document = await document_loader.load_document(DOCUMENT_URL)
        
        print(f"   Document ID after loading: {document.id}")
        print(f"   Document URL: {document.url}")
        print(f"   Document content length: {len(document.content) if hasattr(document, 'content') else 'N/A'}")
        
        # Step 2: Check what's in vector store with this document ID
        print(f"\n🔍 Step 2: Checking vector store for document ID: {document.id}")
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        # Generate a test query
        test_query = "What is the sum insured?"
        query_embedding = await embedding_service.generate_single_embedding(test_query)
        
        # Search with the document ID from the loaded document
        search_results = await vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            document_id=str(document.id),  # Convert to string like in main.py
            top_k=5
        )
        
        print(f"   Found {len(search_results)} results with document ID filter")
        
        if search_results:
            print("   ✅ Document ID matches - chunks found!")
            for i, result in enumerate(search_results[:2]):
                print(f"     Result {i+1}: {result.content[:80]}...")
        else:
            print("   ❌ No results found with this document ID")
            
            # Try searching without filter to see what document IDs exist
            print("\n🔍 Checking what document IDs exist in vector store...")
            all_results = await vector_store.search_similar_chunks(
                query_embedding=query_embedding,
                top_k=5
            )
            
            if all_results:
                existing_doc_ids = set()
                for result in all_results:
                    doc_id = result.document_metadata.get('document_id')
                    if doc_id:
                        existing_doc_ids.add(doc_id)
                
                print(f"   Existing document IDs in vector store:")
                for doc_id in existing_doc_ids:
                    print(f"     - {doc_id}")
                
                print(f"\n   Loaded document ID: {document.id}")
                print(f"   Document ID type: {type(document.id)}")
                
                # Check if any existing ID matches when converted to string
                str_doc_id = str(document.id)
                if str_doc_id in existing_doc_ids:
                    print("   ✅ String conversion matches!")
                else:
                    print("   ❌ No match even with string conversion")
        
        # Step 3: Test the actual API call to see what document ID it uses
        print(f"\n🌐 Step 3: Testing API call...")
        
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json"
        }
        
        request_data = {
            "documents": DOCUMENT_URL,
            "questions": ["What is the sum insured under this policy?"]
        }
        
        # Make the API call and check logs
        print("   Making API request...")
        response = requests.post(
            "http://127.0.0.1:8000/hackrx/run", 
            headers=headers, 
            json=request_data, 
            timeout=30
        )
        
        print(f"   API Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get("answers", [])
            print(f"   Number of answers: {len(answers)}")
            
            if answers:
                answer = answers[0]
                print(f"   Answer preview: {answer[:100]}...")
                
                # Check if it's a generic response
                is_generic = "don't have sufficient information" in answer.lower()
                print(f"   Is generic response: {is_generic}")
                
                if is_generic:
                    print("   ❌ API is returning generic response - document ID mismatch likely")
                else:
                    print("   ✅ API found relevant content!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trace_document_processing())