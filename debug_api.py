import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_id = os.getenv("LLM_MODEL")

print(f"Testing Model: {model_id}")
print(f"API Key found: {bool(api_key)}")

if not api_key or not model_id:
    print("❌ Missing configuration in .env")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_id)

try:
    print("⏳ Sending test query...")
    response = model.generate_content("Hello, this is a test.")
    print("✅ Success!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nAttempting to list available models for this key:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except:
        print("Could not list models.")
