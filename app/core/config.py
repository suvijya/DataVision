"""
Core configuration for PyData Assistant Backend.
Handles environment variables and app settings.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # API Configuration
    PROJECT_NAME: str = "PyData Assistant Backend"
    VERSION: str = "2.0"
    DEBUG: bool = True
    
    # Google Gemini API
    GEMINI_API_KEY: str = ""
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 16777216  # 16MB
    ALLOWED_EXTENSIONS: List[str] = [".csv"]
    
    # Cache Configuration
    CACHE_DIR: str = "cache"
    SESSION_TIMEOUT: int = 86400  # 24 hours
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # For development
        "*"  # Allow all for development
    ]
    
    # LLM Configuration
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    
    # Memory Configuration
    CONVERSATION_MEMORY_LIMIT: int = 2000  # tokens
    MAX_CONVERSATION_MESSAGES: int = 50
    
    # Execution Safety
    CODE_EXECUTION_TIMEOUT: int = 30  # seconds
    MAX_MEMORY_USAGE: int = 536870912  # 512MB
    
    # Allowed Python modules for code execution
    ALLOWED_IMPORTS: List[str] = [
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
    
    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"
        env_ignore_empty = True

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