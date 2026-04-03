# Simple script to activate venv and start the PyData Assistant server
Write-Host "🚀 Preparing PyData Assistant..." -ForegroundColor Cyan

if (-Not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "🔌 Activating virtual environment and installing dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Write-Host "🔥 Starting server..." -ForegroundColor Green
python start_server.py
