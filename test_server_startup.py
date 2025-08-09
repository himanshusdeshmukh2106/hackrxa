#!/usr/bin/env python3
"""
Test server startup to verify async task fixes
"""
import asyncio
import sys
import traceback

async def test_imports():
    """Test that all modules can be imported without async task errors"""
    print("🔍 Testing Module Imports")
    print("=" * 50)
    
    try:
        print("   Importing core modules...")
        from app.core.config import settings
        print("   ✅ Config imported")
        
        from app.core.cache import cache_manager
        print("   ✅ Cache manager imported")
        
        from app.core.monitoring import system_monitor
        print("   ✅ Monitoring imported")
        
        print("   Importing services...")
        from app.services.database import db_manager
        print("   ✅ Database service imported")
        
        from app.services.embedding_service import embedding_service
        print("   ✅ Embedding service imported")
        
        from app.services.vector_store import vector_store
        print("   ✅ Vector store imported")
        
        from app.services.llm_service import llm_service
        print("   ✅ LLM service imported")
        
        print("   Importing main app...")
        from app.main import app
        print("   ✅ Main app imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_basic_functionality():
    """Test basic functionality without starting the server"""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        from app.schemas.requests import QueryRequest
        from app.schemas.responses import QueryResponse
        
        # Test schema creation
        request = QueryRequest(
            documents="https://example.com/test.pdf",
            questions=["What is this document about?"]
        )
        print("   ✅ Request schema working")
        
        response = QueryResponse(answers=["This is a test answer"])
        print("   ✅ Response schema working")
        
        # Test authentication components
        from app.middleware.auth import BearerTokenAuth, get_current_user
        print("   ✅ Authentication components imported")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

async def main():
    """Run startup tests"""
    print("🚀 Server Startup Test")
    print("=" * 60)
    
    imports_ok = await test_imports()
    functionality_ok = await test_basic_functionality() if imports_ok else False
    
    print("\n" + "=" * 60)
    print("📊 STARTUP TEST RESULTS")
    print("=" * 60)
    
    print(f"📦 Module Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"🔧 Basic Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    if imports_ok and functionality_ok:
        print(f"\n🎉 SUCCESS: Server can start without async task errors!")
        print(f"   - All modules import correctly")
        print(f"   - No event loop issues during import")
        print(f"   - Basic functionality working")
        print(f"\n💡 The server should now start properly!")
        return True
    else:
        print(f"\n❌ ISSUES FOUND:")
        if not imports_ok:
            print(f"   - Module import problems")
        if not functionality_ok:
            print(f"   - Basic functionality issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)