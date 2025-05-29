#!/bin/bash

echo "🚀 Starting CLaiM Frontend..."

# Navigate to project root
cd "$(dirname "$0")/.." || exit

# Navigate to frontend
cd frontend || exit

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📥 Installing dependencies..."
    npm install
fi

# Start the frontend
echo "🎯 Starting Vite dev server..."
echo "   Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"

npm run dev