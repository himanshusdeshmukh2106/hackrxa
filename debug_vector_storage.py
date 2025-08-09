#!/usr/bin/env python3
"""
Debug script to check what's actually stored in Pinecone
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.core.config import settings

async def debug_vector_storage():
    """Debug what's stored in the vector database"""
    print("🔍 VECTOR STORAGE DEBUG")
    print("=" * 50)
    
    try:
        # Initialize services
        print("📡 Initializing vector store...")
        await vector_store.initialize()
        
        # Get index stats
        print("\n📊 Index Statistics:")
        stats = await vector_store.get_index_stats()
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Dimension: {stats.get('dimension', 0)}")
        print(f"   Index fullness: {stats.get('index_fullness', 0)}")
        print(f"   Namespaces: {stats.get('namespaces', {})}")
        
        # Test a simple query to see what's returned
        print("\n🔍 Testing sample query...")
        
        # Generate a test embedding
        test_query = "What is the sum insured?"
        print(f"   Query: {test_query}")
        
        await embedding_service.initialize()
        query_embedding = await embedding_service.generate_single_embedding(test_query)
        print(f"   Generated embedding dimension: {len(query_embedding)}")
        
        # Search without filters first
        print("\n🔍 Searching without filters...")
        matches = await vector_store.similarity_search(
            query_vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        print(f"   Found {len(matches)} matches")
        
        for i, match in enumerate(matches):
            print(f"\n   Match {i+1}:")
            print(f"     ID: {match.id}")
            print(f"     Score: {match.score:.4f}")
            print(f"     Metadata keys: {list(match.metadata.keys())}")
            
            # Check document_id in metadata
            doc_id = match.metadata.get('document_id')
            print(f"     Document ID: {doc_id}")
            
            # Show content preview
            content = match.metadata.get('content', '')
            print(f"     Content preview: {content[:100]}...")
        
        # Now test with a document_id filter
        if matches:
            test_doc_id = matches[0].metadata.get('document_id')
            if test_doc_id:
                print(f"\n🔍 Testing with document_id filter: {test_doc_id}")
                
                filtered_matches = await vector_store.similarity_search(
                    query_vector=query_embedding,
                    top_k=5,
                    filter_dict={"document_id": {"$eq": test_doc_id}},
                    include_metadata=True
                )
                
                print(f"   Found {len(filtered_matches)} filtered matches")
                
                for i, match in enumerate(filtered_matches):
                    print(f"     Match {i+1}: {match.id} (score: {match.score:.4f})")
        
        # Test search_similar_chunks method
        print(f"\n🔍 Testing search_similar_chunks method...")
        search_results = await vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=3
        )
        
        print(f"   Found {len(search_results)} search results")
        for i, result in enumerate(search_results):
            print(f"     Result {i+1}:")
            print(f"       Chunk ID: {result.chunk_id}")
            print(f"       Similarity: {result.similarity_score:.4f}")
            print(f"       Content: {result.content[:100]}...")
            print(f"       Doc metadata: {result.document_metadata}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_vector_storage())