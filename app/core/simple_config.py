"""
Simple configuration for PyData Assistant Backend.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Simple settings class without Pydantic complexity."""
    
    # API Configuration
    PROJECT_NAME = "PyData Assistant Backend"
    VERSION = "2.0"
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # File Upload Settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "16777216"))  # 16MB
    ALLOWED_EXTENSIONS = [".csv"]
    
    # Cache Configuration
    CACHE_DIR = os.getenv("CACHE_DIR", "cache")
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "86400"))  # 24 hours
    
    # CORS Configuration
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000", 
        "http://localhost:3000",
        "*"
    ]
    
    # LLM Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    # Memory Configuration
    CONVERSATION_MEMORY_LIMIT = int(os.getenv("CONVERSATION_MEMORY_LIMIT", "2000"))
    MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "50"))
    
    # Execution Safety
    CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))
    MAX_MEMORY_USAGE = int(os.getenv("MAX_MEMORY_USAGE", "536870912"))  # 512MB
    
    # Allowed Python modules for code execution
    ALLOWED_IMPORTS = [
        "pandas",
        "numpy", 
        "plotly.express",
        "plotly.graph_objects",
        "plotly.figure_factory",
        "scipy.stats",
        "scipy.cluster",
        "sklearn.preprocessing",
        "sklearn.cluster",
        "sklearn.decomposition"
    ]

# Global settings instance
settings = Settings()

def validate_configuration():
    """Validate that required configuration is present."""
    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is required. Please set it in your .env file.\n"
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )
    
    # Create cache directory if it doesn't exist
    os.makedirs(settings.CACHE_DIR, exist_ok=True)
    
    print(f"✅ Configuration validated")
    print(f"   Project: {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"   Debug: {settings.DEBUG}")
    print(f"   Cache: {settings.CACHE_DIR}")
    print(f"   Model: {settings.LLM_MODEL}")