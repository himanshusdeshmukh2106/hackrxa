#!/usr/bin/env python3
"""
Comprehensive debug script to trace the entire document processing pipeline
"""
import requests
import json
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.document_loader import DocumentLoader
from app.services.text_extractor import TextExtractor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store, Vector
from app.services.search_engine import search_engine

# Document URL
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

async def debug_full_pipeline():
    """Debug the entire document processing pipeline step by step"""
    print("🔍 FULL PIPELINE DEBUG")
    print("=" * 60)
    
    try:
        # Step 1: Load document
        print("📄 Step 1: Loading document...")
        async with DocumentLoader() as document_loader:
            document = await document_loader.load_document(DOCUMENT_URL)
        
        print(f"   ✅ Document loaded")
        print(f"   Document ID: {document.id}")
        print(f"   Document type: {document.document_type}")
        print(f"   Status: {document.status}")
        
        # Step 2: Extract text
        print(f"\n📝 Step 2: Extracting text...")
        text_extractor = TextExtractor()
        text_chunks = await text_extractor.extract_text(document)
        print(f"   ✅ Text extracted")
        print(f"   Number of chunks: {len(text_chunks)}")
        
        if text_chunks:
            print(f"   First chunk preview: {text_chunks[0].content[:100]}...")
            print(f"   Last chunk preview: {text_chunks[-1].content[:100]}...")
        
        # Step 3: Generate embeddings
        print(f"\n🧠 Step 3: Generating embeddings...")
        await embedding_service.initialize()
        
        embedded_chunks = await embedding_service.embed_text_chunks(text_chunks[:5])  # Test with first 5 chunks
        print(f"   ✅ Embeddings generated")
        print(f"   Embedded chunks: {len(embedded_chunks)}")
        
        chunks_with_embeddings = [chunk for chunk in embedded_chunks if chunk.embedding]
        print(f"   Chunks with embeddings: {len(chunks_with_embeddings)}")
        
        if chunks_with_embeddings:
            print(f"   Embedding dimension: {len(chunks_with_embeddings[0].embedding)}")
        
        # Step 4: Store in vector database
        print(f"\n🗄️ Step 4: Storing in vector database...")
        await vector_store.initialize()
        
        # Convert to Vector objects
        vectors = [
            Vector(
                id=chunk.id,
                values=chunk.embedding,
                metadata={
                    "document_id": chunk.document_id,
                    "content": chunk.content[:500],
                    "chunk_index": chunk.chunk_index
                }
            ) for chunk in chunks_with_embeddings
        ]
        
        success = await vector_store.upsert_vectors_fast(vectors)
        print(f"   ✅ Storage result: {success}")
        
        # Wait a moment for indexing
        await asyncio.sleep(2)
        
        # Step 5: Test search
        print(f"\n🔍 Step 5: Testing search...")
        test_query = "What is the sum insured under this policy?"
        
        query_embedding = await embedding_service.generate_single_embedding(test_query)
        print(f"   Query: {test_query}")
        print(f"   Query embedding dimension: {len(query_embedding)}")
        
        # Search without filter
        print(f"\n   🔍 Search without document filter...")
        search_results_no_filter = await vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=3
        )
        print(f"   Results without filter: {len(search_results_no_filter)}")
        
        for i, result in enumerate(search_results_no_filter):
            print(f"     Result {i+1}: Score {result.similarity_score:.4f}")
            print(f"       Content: {result.content[:80]}...")
            print(f"       Doc ID: {result.document_metadata.get('document_id')}")
        
        # Search with document filter
        print(f"\n   🔍 Search with document filter: {document.id}")
        search_results_filtered = await vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            document_id=str(document.id),
            top_k=3
        )
        print(f"   Results with filter: {len(search_results_filtered)}")
        
        for i, result in enumerate(search_results_filtered):
            print(f"     Result {i+1}: Score {result.similarity_score:.4f}")
            print(f"       Content: {result.content[:80]}...")
        
        # Step 6: Test search engine
        print(f"\n🔎 Step 6: Testing search engine...")
        await search_engine.initialize() if hasattr(search_engine, 'initialize') else None
        
        search_engine_results = await search_engine.hybrid_search(
            query=test_query,
            filters={"document_id": str(document.id)},
            top_k=3
        )
        print(f"   Search engine results: {len(search_engine_results)}")
        
        for i, result in enumerate(search_engine_results):
            print(f"     Result {i+1}: Score {result.similarity_score:.4f}")
            print(f"       Content: {result.content[:80]}...")
        
        # Step 7: Check vector store stats
        print(f"\n📊 Step 7: Vector store statistics...")
        stats = await vector_store.get_index_stats()
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Index fullness: {stats.get('index_fullness', 0)}")
        
        # Summary
        print(f"\n📋 PIPELINE SUMMARY:")
        print(f"   Document loaded: ✅")
        print(f"   Text extracted: ✅ ({len(text_chunks)} chunks)")
        print(f"   Embeddings generated: ✅ ({len(chunks_with_embeddings)} embedded)")
        print(f"   Vectors stored: ✅ ({success})")
        print(f"   Search without filter: {'✅' if search_results_no_filter else '❌'} ({len(search_results_no_filter)} results)")
        print(f"   Search with filter: {'✅' if search_results_filtered else '❌'} ({len(search_results_filtered)} results)")
        print(f"   Search engine: {'✅' if search_engine_results else '❌'} ({len(search_engine_results)} results)")
        
        if not search_results_filtered:
            print(f"\n❌ PROBLEM IDENTIFIED: Document filter not working!")
            print(f"   Expected document ID: {document.id}")
            print(f"   Stored document IDs: {set(r.document_metadata.get('document_id') for r in search_results_no_filter)}")
        elif not search_engine_results:
            print(f"\n❌ PROBLEM IDENTIFIED: Search engine not finding results!")
        else:
            print(f"\n✅ PIPELINE WORKING: All steps successful!")
        
    except Exception as e:
        print(f"❌ Pipeline debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_full_pipeline())