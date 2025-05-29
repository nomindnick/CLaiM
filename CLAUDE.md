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
python scripts/convert_to_gguf.py  # Convert models to GGUF format
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
- `frontend/src/components/PrivacyIndicator.tsx` - Privacy UI component

## Development Plan & Status

### Current Sprint (Phase 1 - Foundation)
1. ⬜ Project structure and documentation (IN PROGRESS)
2. ⬜ Document processor module 
   - ⬜ PDF boundary detection using pymupdf
   - ⬜ DistilBERT integration for classification
   - ⬜ Basic metadata extraction
3. ⬜ Storage module with SQLite
4. ⬜ Basic web UI with document browser

### Implementation Order
1. **Document Processor** → Start here, most complex module
2. **Storage** → Required by all other modules 
3. **Search** → Enables basic usability
4. **AI Interface** → Adds intelligence layer
5. **Graph Engine** → Advanced analysis features
6. **Financial/Timeline** → Specialized features
7. **Electron packaging** → Production deployment

### Immediate Next Steps
1. Initial project structure setup, git initialization, other preliminary development tasks 
2. Implement `PDFSplitter` class in `backend/modules/document_processor/pdf_splitter.py`
3. Create sample construction PDFs for testing (RFIs, change orders, emails)
4. Set up DistilBERT model loading in `backend/modules/ai_classifier/`
5. Design document metadata schema in Pydantic models
6. Implement basic FastAPI routes for document upload

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
- [2024-01-XX] Set up initial project structure
- [2024-01-XX] Created comprehensive documentation
- [2024-01-XX] Defined data models and API structure

## Debug Commands
```bash
# Check model loading
python -c "from backend.modules.ai_interface.model_manager import ModelManager; ModelManager().status()"

# Test document processing
python scripts/test_document_processor.py sample_data/test_rfi.pdf

# Verify database setup
python -c "from backend.modules.storage.sqlite_handler import test_connection; test_connection()"
```

## References
- Internal specs: `claim-project-structure.md` and 'claim-specification.md'

## Notes for Claude Code
- This is a legal tool - accuracy and auditability are paramount
- Always handle errors gracefully with user-friendly messages
- When in doubt, fail safely and log details for debugging
- Remember the users are attorneys, not software engineers
- Test with realistic construction litigation scenarios
- Update this file's "Recently Completed Tasks" section after major work
