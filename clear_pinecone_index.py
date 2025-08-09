#!/usr/bin/env python3
"""
Clear Pinecone index to allow re-processing with deterministic document IDs
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.vector_store import vector_store

async def clear_index():
    """Clear all vectors from Pinecone index"""
    print("🧹 CLEARING PINECONE INDEX")
    print("=" * 40)
    
    try:
        # Initialize vector store
        await vector_store.initialize()
        
        # Get current stats
        stats = await vector_store.get_index_stats()
        current_count = stats.get('total_vector_count', 0)
        
        print(f"📊 Current vector count: {current_count}")
        
        if current_count == 0:
            print("✅ Index is already empty")
            return
        
        # Clear the index by deleting all vectors
        print("🗑️ Clearing all vectors...")
        
        # Use the sync client to delete all vectors
        index = vector_store.sync_client.Index(vector_store.index_name)
        index.delete(delete_all=True)
        
        print("✅ Index cleared successfully!")
        
        # Wait a moment and check stats
        await asyncio.sleep(2)
        new_stats = await vector_store.get_index_stats()
        new_count = new_stats.get('total_vector_count', 0)
        
        print(f"📊 New vector count: {new_count}")
        
        if new_count == 0:
            print("🎉 Index successfully cleared!")
        else:
            print(f"⚠️ {new_count} vectors still remain (may take time to update)")
        
    except Exception as e:
        print(f"❌ Failed to clear index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("⚠️ WARNING: This will delete all vectors from Pinecone!")
    print("The system will need to re-process documents after this.")
    print()
    
    confirm = input("Are you sure you want to continue? (yes/no): ").lower().strip()
    
    if confirm == 'yes':
        asyncio.run(clear_index())
    else:
        print("❌ Operation cancelled")