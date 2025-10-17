#!/usr/bin/env python3
"""
Test script to check Gemini API functionality
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini_api():
    api_key = os.getenv('GEMINI_API_KEY', '')
    
    if not api_key:
        print("❌ No GEMINI_API_KEY found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI library imported")
        
        genai.configure(api_key=api_key)
        print("✅ API configured")
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Model created")
        
        response = model.generate_content("Hello, this is a test. Please respond with 'API Working'")
        print(f"✅ API Response: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

def test_pandas_functionality():
    try:
        import pandas as pd
        import io
        
        print("✅ Pandas imported")
        
        # Test CSV reading
        csv_data = """name,age,city
John,25,New York
Jane,30,Los Angeles"""
        
        df = pd.read_csv(io.StringIO(csv_data))
        print(f"✅ CSV reading test: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing pandas: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PyData Assistant Components\n")
    
    print("1. Testing Pandas functionality:")
    pandas_ok = test_pandas_functionality()
    
    print("\n2. Testing Gemini API:")
    gemini_ok = test_gemini_api()
    
    print(f"\n📊 Results:")
    print(f"   Pandas: {'✅ Working' if pandas_ok else '❌ Failed'}")
    print(f"   Gemini API: {'✅ Working' if gemini_ok else '❌ Failed'}")
    
    if pandas_ok and gemini_ok:
        print("\n🎉 All components working! You can run the application.")
    else:
        print("\n⚠️  Some components need attention before running the application.")