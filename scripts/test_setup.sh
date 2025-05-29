#!/bin/bash

echo "🧪 Testing CLaiM Setup"
echo "===================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is open
port_is_open() {
    nc -z localhost "$1" 2>/dev/null
}

echo -e "\n📋 Checking Dependencies:"
echo "------------------------"

# Check Python
if command_exists python3; then
    echo -e "${GREEN}✓${NC} Python3 installed: $(python3 --version)"
else
    echo -e "${RED}✗${NC} Python3 not found"
fi

# Check Node
if command_exists node; then
    echo -e "${GREEN}✓${NC} Node.js installed: $(node --version)"
else
    echo -e "${RED}✗${NC} Node.js not found"
fi

# Check npm
if command_exists npm; then
    echo -e "${GREEN}✓${NC} npm installed: $(npm --version)"
else
    echo -e "${RED}✗${NC} npm not found"
fi

# Check tesseract
if command_exists tesseract; then
    echo -e "${GREEN}✓${NC} Tesseract installed: $(tesseract --version | head -1)"
else
    echo -e "${YELLOW}⚠${NC} Tesseract not found (required for OCR)"
fi

echo -e "\n🐍 Checking Python Environment:"
echo "-------------------------------"

# Check virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found"
    
    # Activate and check packages
    source venv/bin/activate
    
    # Check critical packages
    for pkg in fastapi uvicorn pydantic sqlalchemy pymupdf; do
        if python -c "import $pkg" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $pkg installed"
        else
            echo -e "${RED}✗${NC} $pkg not installed"
        fi
    done
else
    echo -e "${RED}✗${NC} Virtual environment not found"
fi

echo -e "\n🌐 Checking Services:"
echo "--------------------"

# Check if backend is running
if port_is_open 8000; then
    echo -e "${GREEN}✓${NC} Backend running on port 8000"
    
    # Test API endpoint
    response=$(curl -s http://localhost:8000/health)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} API health check passed"
    else
        echo -e "${RED}✗${NC} API health check failed"
    fi
else
    echo -e "${YELLOW}⚠${NC} Backend not running on port 8000"
    echo "  To start: cd backend && uvicorn api.main:app --reload"
fi

# Check if frontend is running
if port_is_open 5173; then
    echo -e "${GREEN}✓${NC} Frontend running on port 5173"
else
    echo -e "${YELLOW}⚠${NC} Frontend not running on port 5173"
    echo "  To start: cd frontend && npm run dev"
fi

echo -e "\n📁 Checking Required Files:"
echo "---------------------------"

# Check backend files
for file in "backend/.env" "backend/api/main.py" "backend/requirements.txt"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
    fi
done

# Check frontend files
for file in "frontend/package.json" "frontend/src/App.tsx"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
    fi
done

echo -e "\n🚀 Quick Start Commands:"
echo "-----------------------"
echo "1. Start Backend:"
echo "   cd backend && source ../venv/bin/activate && uvicorn api.main:app --reload"
echo ""
echo "2. Start Frontend:"
echo "   cd frontend && npm run dev"
echo ""
echo "3. Access Application:"
echo "   Frontend: http://localhost:5173"
echo "   API Docs: http://localhost:8000/api/v1/docs"

echo -e "\n✅ Setup check complete!"