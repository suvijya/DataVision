"""
PyData Assistant Backend - Main FastAPI Application
====================================================

AI-powered data analysis platform that democratizes data science
by allowing users to analyze CSV datasets using natural language queries.

Author: PyData Assistant Team
Version: 2.0
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
import numpy as np

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.simple_config import settings, validate_configuration
from app.api.v1.endpoints.session import router as session_router
from app.api.v1.endpoints.statistical_analysis import router as statistical_analysis_router

# Custom JSON encoder for NumPy types
class NumpyJSONEncoder:
    """Custom JSON encoder to handle NumPy types."""
    
    @staticmethod
    def convert_numpy(obj):
        """Convert NumPy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    @staticmethod
    def convert_dict_numpy(data):
        """Recursively convert NumPy types in nested data structures."""
        if isinstance(data, dict):
            return {k: NumpyJSONEncoder.convert_dict_numpy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [NumpyJSONEncoder.convert_dict_numpy(item) for item in data]
        else:
            return NumpyJSONEncoder.convert_numpy(data)


def custom_jsonable_encoder(obj):
    """Custom jsonable encoder that handles NumPy types."""
    # First convert NumPy types
    converted = NumpyJSONEncoder.convert_dict_numpy(obj)
    # Then use FastAPI's jsonable_encoder
    return jsonable_encoder(converted)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting PyData Assistant Backend...")
    try:
        validate_configuration()
        logger.info("✅ Startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PyData Assistant Backend...")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="""
    🤖 **AI-Powered Data Analysis Platform**
    
    Transform your CSV data into insights using natural language queries.
    
    ## Features
    * 📊 **Smart CSV Analysis** - Upload and get instant insights
    * 🤖 **AI-Powered Queries** - Ask questions in plain English  
    * 📈 **Interactive Visualizations** - Plotly charts generated automatically
    * 💬 **Conversational Interface** - ChatGPT-like experience
    * 🔒 **Secure Execution** - Sandboxed code execution
    * 💾 **Session Management** - Persistent analysis sessions
    
    ## Quick Start
    1. Upload your CSV file using `/api/v1/session/start`
    2. Query your data using `/api/v1/session/query`
    3. Get insights, charts, and statistics instantly!
    
    ## Example Queries
    * "Show me sales by region"
    * "Create a histogram of customer age"
    * "What's the correlation between price and sales?"
    * "Identify trends in revenue over time"
    """,
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    session_router,
    prefix="/api/v1",
    tags=["sessions"]
)

app.include_router(
    statistical_analysis_router,
    prefix="/api/v1",
    tags=["statistical-analysis"]
)

# Mount frontend static files
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    
# Also mount frontend files directly for easier access
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "debug": settings.DEBUG,
        "timestamp": "2025-10-17T14:00:00Z"
    }

# Root endpoint - serve frontend
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application."""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>PyData Assistant</title></head>
                <body style="font-family: system-ui; padding: 2rem; text-align: center;">
                    <h1>🤖 PyData Assistant Backend</h1>
                    <p>Backend is running successfully!</p>
                    <p>Frontend not found. Please ensure frontend/index.html exists.</p>
                    <p><a href="/docs">📚 View API Documentation</a></p>
                </body>
            </html>
            """
        )

# App route - serve frontend
@app.get("/app", response_class=HTMLResponse)
async def app_route():
    """Alternative route to serve the main application."""
    return await root()

# Serve CSS file
@app.get("/styles.css")
async def get_styles():
    """Serve the CSS file."""
    css_path = "frontend/styles.css"
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

# Serve JS file
@app.get("/script.js")
async def get_script():
    """Serve the JavaScript file."""
    js_path = "frontend/script.js"
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="text/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if settings.DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc),
                "type": type(exc).__name__,
                "request_url": str(request.url)
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
            }
        )

if __name__ == "__main__":
    """Run the application directly."""
    print(f"""
🤖 PyData Assistant Backend v{settings.VERSION}
==========================================

Starting server...
Frontend: http://localhost:8000/
API Docs: http://localhost:8000/docs
Health:   http://localhost:8000/health

Press Ctrl+C to stop
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )