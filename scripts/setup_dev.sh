#!/bin/bash
# Development environment setup script for CLaiM

set -e

echo "🚀 Setting up CLaiM development environment..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "Error: Python 3.11+ is required. Current version: $python_version"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install backend dependencies
echo -e "${YELLOW}Installing backend dependencies...${NC}"
cd backend
pip install -r requirements-dev.txt
cd ..
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Check Node.js
echo -e "${YELLOW}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js 18+"
    exit 1
fi
node_version=$(node --version)
echo -e "${GREEN}✓ Node.js $node_version${NC}"

# Install frontend dependencies
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
cd frontend
npm install
cd ..
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p backend/uploads
mkdir -p backend/logs
mkdir -p models
echo -e "${GREEN}✓ Directories created${NC}"

# Copy environment file
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp backend/.env.example backend/.env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠️  Please update backend/.env with your configuration${NC}"
fi

# Initialize git hooks (if pre-commit is available)
if command -v pre-commit &> /dev/null; then
    echo -e "${YELLOW}Setting up pre-commit hooks...${NC}"
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi

echo -e "\n${GREEN}✅ Development environment setup complete!${NC}"
echo -e "\n${YELLOW}To start developing:${NC}"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start backend: cd backend && uvicorn api.main:app --reload"
echo "3. Start frontend: cd frontend && npm run dev"
echo -e "\n${YELLOW}Other useful commands:${NC}"
echo "- Run tests: pytest"
echo "- Format code: black backend/"
echo "- Type check: mypy backend/"
echo "- Lint frontend: cd frontend && npm run lint"