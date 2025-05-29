# CLaiM - Construction Litigation AI Manager

Transform construction litigation documents into actionable intelligence with privacy-first AI analysis.

## Overview

CLaiM is a desktop application designed for attorneys handling construction disputes. It processes large PDF files containing mixed document types (emails, RFIs, change orders, invoices, etc.) and provides AI-powered analysis while maintaining complete privacy control.

### Key Features

- **Intelligent Document Splitting**: Automatically identifies and extracts individual documents from large PDFs
- **AI-Powered Classification**: Categorizes documents by type using fine-tuned DistilBERT
- **Privacy-First Design**: Three-tier privacy model - from fully local to hybrid cloud processing
- **Advanced Search**: Hybrid keyword + semantic search across all documents
- **Claim Analysis**: Maps relationships between claims and supporting evidence
- **Financial Tracking**: Reconciles payments, change orders, and disputed amounts

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nomindnick/CLaiM.git
cd CLaiM
```

2. Run the setup script:
```bash
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh
```

This will:
- Create a Python virtual environment
- Install all backend dependencies
- Install all frontend dependencies
- Create necessary directories
- Set up the development environment

3. Copy and configure the environment file:
```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your settings
```

### Running the Application

1. Start the backend API:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd backend
uvicorn api.main:app --reload
```

The API will be available at http://localhost:8000
API documentation: http://localhost:8000/api/v1/docs

2. Start the frontend (in a new terminal):
```bash
cd frontend
npm run dev
```

The frontend will be available at http://localhost:5173

## Development

### Project Structure

```
CLaiM/
├── backend/           # FastAPI backend
│   ├── api/          # API configuration and main app
│   ├── modules/      # Feature modules
│   └── shared/       # Shared utilities
├── frontend/         # React TypeScript frontend
│   ├── src/          # Source code
│   └── public/       # Static assets
├── models/           # AI model storage (GGUF format)
├── scripts/          # Development and deployment scripts
└── docs/            # Additional documentation
```

### Running Tests

Backend tests:
```bash
cd backend
pytest
# With coverage
pytest --cov=backend --cov-report=html
```

Frontend tests:
```bash
cd frontend
npm test
```

### Code Quality

Format Python code:
```bash
black backend/
isort backend/
```

Type check Python:
```bash
mypy backend/
```

Lint frontend:
```bash
cd frontend
npm run lint
```

## Privacy Modes

CLaiM offers three privacy modes:

1. **Full Local** 🔒: All processing happens on your computer
2. **Hybrid Safe** 🔐: Core operations local, only non-sensitive analysis uses APIs
3. **Full Featured** 🌐: Cloud APIs used for enhanced capabilities

## AI Models

CLaiM uses the following models:

- **DistilBERT** (250MB): Document classification and metadata extraction
- **Phi-3.5-mini** (2GB): Text generation and complex analysis
- **MiniLM** (90MB): Semantic embeddings for search

Download models:
```bash
python scripts/download_models.py
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## Contributing

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make your changes and add tests
3. Run tests and ensure they pass
4. Commit using conventional commits: `git commit -m "feat: add new feature"`
5. Push and create a pull request

## License

[MIT License](LICENSE)

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/nomindnick/CLaiM/issues).

## Roadmap

- [ ] Phase 1: Document processing and classification (In Progress)
- [ ] Phase 2: AI integration and search
- [ ] Phase 3: Financial analysis and timeline building
- [ ] Phase 4: Desktop packaging with Electron

See [CLAUDE.md](CLAUDE.md) for detailed development instructions and project specifications.