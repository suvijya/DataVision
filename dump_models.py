import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY', '')
if not api_key:
    print("❌ No GEMINI_API_KEY found")
    exit(1)

genai.configure(api_key=api_key)

with open('available_models.txt', 'w') as f:
    try:
        models = genai.list_models()
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                f.write(f"{m.name}\n")
        print("✅ Models saved to available_models.txt")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
