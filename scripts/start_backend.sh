#!/bin/bash

echo "🚀 Starting CLaiM Backend..."

# Navigate to project root
cd "$(dirname "$0")/.." || exit

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Navigate to backend
cd backend || exit

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install -r requirements-minimal.txt
fi

# Start the backend
echo "🎯 Starting FastAPI server..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/api/v1/docs"
echo ""
echo "Press Ctrl+C to stop"

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000