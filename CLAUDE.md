# CLaiM - Construction Litigation AI Manager

## Project Overview
Desktop application for analyzing construction litigation documents, specifically focused on California public agency (school district) construction disputes. The system processes large PDFs, extracts individual documents, and provides AI-powered analysis for case preparation.

**IMPORTANT**: This is built for non-technical attorneys. Prioritize user experience and clear error messages over technical elegance.

## California School Construction Context
- Public works projects require prevailing wages (Labor Code 1770+)
- Public Contract Code governs bidding and contracts 
- DSA (Division of State Architect) approval required for school buildings
- Change orders over certain thresholds need board approval
- Strict notice requirements for claims (Government Code 910-946.6)
- Documentary evidence is critical for delay/disruption claims

## Quick Start (Current State)

```bash
# First-time setup
./scripts/setup_dev.sh
cp backend/.env.example backend/.env

# Start backend (Terminal 1)
source venv/bin/activate
cd backend
uvicorn api.main:app --reload

# Start frontend (Terminal 2)
cd frontend
npm run dev

# Access application
# Frontend: http://localhost:5173
# API Docs: http://localhost:8000/api/v1/docs
```

## Common Commands

### Development
```bash
# Start development environment
source venv/bin/activate  # or venv\Scripts\activate on Windows
cd backend && uvicorn api.main:app --reload  # Start API on :8000
cd frontend && npm run dev  # Start frontend on :5173

# Testing
pytest  # Run all tests
pytest backend/modules/document_processor/tests/  # Test specific module
pytest --cov=backend/modules --cov-report=html  # Coverage report

# Code quality
black backend/  # Format Python code
isort backend/  # Sort imports
mypy backend/  # Type checking
cd frontend && npm run lint  # Lint TypeScript

# Model management
python scripts/download_models.py  # Download required AI models
python scripts/convert_to_gguf.py  # Convert models to GGUF format (not yet implemented)
```

### Git workflow
```bash
git checkout -b feat/module-name  # Create feature branch
git commit -m "feat: add document classification"  # Conventional commits
git push origin feat/module-name  # Push branch
gh pr create  # Create pull request
```

## Core Files & Architecture

### Key Modules
- `backend/modules/document_processor/` - PDF splitting and classification
- `backend/modules/storage/` - SQLite + Qdrant + DuckDB storage layer
- `backend/modules/ai_interface/` - LLM routing and privacy management
- `backend/modules/search/` - Hybrid keyword + semantic search
- `backend/modules/graph_engine/` - Claim-evidence relationship mapping

### Critical Files
- `backend/api/config.py` - Global configuration and environment settings
- `backend/api/privacy_manager.py` - Privacy mode implementation
- `backend/shared/exceptions.py` - Custom exceptions for error handling
- `backend/modules/document_processor/models.py` - Core document data models
- `backend/modules/document_processor/pdf_splitter.py` - PDF processing logic
- `frontend/src/App.tsx` - Main React component with privacy indicator

## Development Plan & Status

### Current Sprint (Phase 1 - Foundation)
1. ✅ Project structure and documentation (COMPLETED)
2. 🟡 Document processor module (IN PROGRESS)
   - ✅ PDF boundary detection using pymupdf
   - ✅ Basic PDF splitting implementation
   - ✅ Document models with Pydantic
   - ✅ FastAPI routes for upload/processing
   - ⬜ OCR integration with pytesseract
   - ⬜ DistilBERT integration for classification
   - ⬜ Advanced metadata extraction
3. ⬜ Storage module with SQLite
4. 🟡 Basic web UI with document browser (STARTED)
   - ✅ React TypeScript setup with Vite
   - ✅ Privacy mode indicator
   - ⬜ Document upload interface
   - ⬜ Document list view

### Implementation Order
1. **Document Processor** → IN PROGRESS - OCR and classification next
2. **Storage** → Required by all other modules 
3. **Search** → Enables basic usability
4. **AI Interface** → Adds intelligence layer
5. **Graph Engine** → Advanced analysis features
6. **Financial/Timeline** → Specialized features
7. **Electron packaging** → Production deployment

### Immediate Next Steps
1. ✅ ~~Initial project structure setup, git initialization~~
2. ✅ ~~Implement `PDFSplitter` class~~
3. ⬜ Implement OCR for scanned pages in `document_processor/ocr_handler.py`
4. ⬜ Create sample construction PDFs for testing
5. ⬜ Implement Storage module with SQLite
   - Document storage schema
   - Full-text search with FTS5
   - CRUD operations
6. ⬜ Set up AI Classifier module structure
   - DistilBERT model loading
   - Document type classification
   - Confidence scoring
7. ⬜ Build document upload UI in frontend
8. ⬜ Connect frontend to backend API

## Testing Strategy
- **Unit tests**: Every public method, aim for 80% coverage
- **Integration tests**: Module interactions, especially storage layer
- **Sample data**: Use realistic construction document examples
- **Performance benchmarks**: Track processing speed for 1000-page PDFs
- **Manual testing**: Always test with real attorney workflows

## Code Style & Standards
- Python 3.11+ with type hints EVERYWHERE
- Pydantic for all data validation (not just dataclasses)
- Descriptive names: `extract_parties_from_document()` not `get_parties()`
- Docstrings with examples for public APIs
- Error messages written for attorneys, not developers
- Max line length: 100 characters

## Module Development Checklist
When creating a new module:
- [ ] Create directory structure per `CLaiM - Project Structure.md`
- [ ] Define Pydantic models with validation
- [ ] Write test cases FIRST (TDD approach)
- [ ] Implement core logic with proper error handling
- [ ] Add FastAPI router with clear endpoints
- [ ] Update `api/main.py` to include router
- [ ] Document all endpoints in docstrings
- [ ] Add module overview to `docs/MODULES.md`
- [ ] Test with realistic data samples

## Common Patterns

### Privacy-Aware Processing
```python
# Always check privacy mode before AI operations
if privacy_mode == PrivacyMode.FULL_LOCAL:
    result = local_model.process(document)
else:
    result = privacy_router.route(document, task)
```

### Document Processing Pipeline
```python
# Standard flow for all documents
document = pdf_splitter.extract(pdf_path)
document = classifier.classify(document)
document = metadata_extractor.extract(document)
embeddings = embedder.embed(document.text)
storage.save(document, embeddings)
```

## Known Issues & Workarounds
- **Large PDF memory usage**: Stream pages, don't load entire PDF
- **OCR accuracy on faxed documents**: Common in construction, use confidence thresholds
- **Table extraction**: Construction schedules are complex, may need manual review
- **Date parsing**: Multiple formats in use, normalize to ISO 8601
- **Party name variations**: "ABC Corp" vs "ABC Corporation" - implement fuzzy matching

## Performance Targets
- Document classification: <100ms per document
- 1000-page PDF processing: <5 minutes total
- Search response: <1 second
- UI responsiveness: <100ms for all interactions

## Model Configuration
- **DistilBERT**: Always loaded, ~250MB, for classification
- **Phi-3.5-mini**: Lazy loaded, ~2GB, for generation tasks
- **MiniLM**: Always loaded, ~90MB, for embeddings
- Models stored in `models/` directory in GGUF format

## Recently Completed Tasks
<!-- Claude Code should update this section after completing work -->
- [2025-05-29] Set up initial project structure with full directory tree
- [2025-05-29] Created comprehensive documentation (README, CLAUDE.md, specs)
- [2025-05-29] Initialized Git repository with .gitignore and LFS support
- [2025-05-29] Set up Python backend with FastAPI, config, and privacy manager
- [2025-05-29] Created React TypeScript frontend with Vite and Tailwind CSS
- [2025-05-29] Implemented Document Processor module foundation:
  - PDF splitter with boundary detection
  - Pydantic models for documents, pages, and metadata
  - FastAPI routes for upload and processing
  - Initial test suite for models
- [2025-05-29] Created development scripts (setup_dev.sh, download_models.py)
- [2025-05-29] Connected to GitHub repository

## Debug Commands
```bash
# Check API health
curl http://localhost:8000/health

# Test document processor
python -m pytest backend/modules/document_processor/tests/ -v

# Check current privacy mode
curl http://localhost:8000/api/v1/privacy

# Future: Check model loading (not yet implemented)
# python -c "from backend.modules.ai_interface.model_manager import ModelManager; ModelManager().status()"

# Future: Test document processing (script not yet created)
# python scripts/test_document_processor.py sample_data/test_rfi.pdf

# Future: Verify database setup (not yet implemented)
# python -c "from backend.modules.storage.sqlite_handler import test_connection; test_connection()"
```

## References
- Internal specs: `claim-project-structure.md` and 'claim-specification.md'

## Current Development Focus

### Priority 1: Complete Document Processor
1. **OCR Handler** (`document_processor/ocr_handler.py`)
   - Integrate pytesseract for scanned page processing
   - Implement confidence scoring
   - Handle mixed text/scanned documents
   
2. **Metadata Extraction** (`metadata_extractor/` module)
   - Extract dates, parties, amounts from documents
   - Use regex patterns for common formats
   - Build party name normalization

### Priority 2: Storage Module
1. **SQLite Schema** (`storage/models.py`)
   - Design document storage tables
   - Implement FTS5 for full-text search
   - Add version tracking for documents
   
2. **Storage Operations** (`storage/sqlite_handler.py`)
   - CRUD operations for documents
   - Batch insert for performance
   - Transaction management

### Priority 3: Frontend Document Browser
1. **Upload Component** (`frontend/src/components/document-browser/`)
   - Drag-and-drop PDF upload
   - Upload progress indicator
   - File validation
   
2. **Document List** 
   - Table view with sorting/filtering
   - Document type badges
   - Quick preview on hover

## Notes for Claude Code
- This is a legal tool - accuracy and auditability are paramount
- Always handle errors gracefully with user-friendly messages
- When in doubt, fail safely and log details for debugging
- Remember the users are attorneys, not software engineers
- Test with realistic construction litigation scenarios
- Update this file's "Recently Completed Tasks" section after major work
- Focus on getting core document processing working end-to-end before adding AI features
