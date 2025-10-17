#!/usr/bin/env python3
"""
List available Gemini models
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY', '')
if not api_key:
    print("❌ No GEMINI_API_KEY found")
    exit(1)

genai.configure(api_key=api_key)

print("📋 Available Gemini Models:\n")

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
            print(f"   Description: {model.description}")
            print(f"   Input limit: {model.input_token_limit}")
            print(f"   Output limit: {model.output_token_limit}")
            print()
except Exception as e:
    print(f"❌ Error listing models: {e}")