#!/usr/bin/env python3
"""
Setup verification script for PyData Assistant
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        'app/main.py',
        'app/core/simple_config.py',
        'app/api/v1/endpoints/session.py',
        'app/services/session_manager.py',
        'app/services/data_analysis.py',
        'frontend/index.html',
        'frontend/styles.css',
        'frontend/script.js',
        '.env',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print("\n❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All required files present")
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_modules = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'plotly',
        'google.generativeai',
        'dotenv',
        'pydantic_settings'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"  ❌ {module}")
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_config():
    """Check configuration"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from app.core.simple_config import settings
        
        if not settings.GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY not configured")
            print("Please add your API key to the .env file")
            return False
        
        print(f"✅ API Key configured: {settings.GEMINI_API_KEY[:10]}...")
        print(f"✅ Model: {settings.LLM_MODEL}")
        print(f"✅ Debug mode: {settings.DEBUG}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_api():
    """Test API functionality"""
    print("\n🧪 Testing API functionality...")
    
    try:
        # Import test function
        sys.path.append('.')
        from test_gemini import test_gemini_api, test_pandas_functionality
        
        pandas_ok = test_pandas_functionality()
        gemini_ok = test_gemini_api()
        
        if pandas_ok and gemini_ok:
            print("✅ API tests passed")
            return True
        else:
            print("❌ API tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def main():
    print("🚀 PyData Assistant Setup Check")
    print("=" * 40)
    
    checks = [
        ("Files", check_files),
        ("Dependencies", check_dependencies), 
        ("Configuration", check_config),
        ("API", test_api)
    ]
    
    all_passed = True
    for name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ Error in {name} check: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! You're ready to run the application.")
        print("\nTo start the server:")
        print("  python app/main.py")
        print("\nThen open: http://localhost:8000/")
    else:
        print("⚠️ Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)