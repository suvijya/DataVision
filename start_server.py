#!/usr/bin/env python3
"""
Simple server starter for PyData Assistant
"""

import os
import sys
import time
import webbrowser
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    print("🌐 Opening browser...")
    webbrowser.open('http://localhost:8000')

def main():
    print("🚀 Starting PyData Assistant Server")
    print("=" * 50)
    
    # Check if all files exist
    required_files = ['app/main.py', 'frontend/index.html', 'frontend/styles.css', 'frontend/script.js']
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing files:")
        for file in missing:
            print(f"   - {file}")
        print("\nPlease ensure all files are in place.")
        return False
    
    print("✅ All required files found")
    print("✅ Starting server on http://localhost:8000")
    print("✅ API documentation at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Start the server
    try:
        import uvicorn
        
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False  # Disable reload to avoid issues
        )
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)