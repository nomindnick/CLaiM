# CLaiM - Construction Litigation AI Manager

## Project Context
Desktop application for California attorneys handling school construction disputes. Processes large PDFs containing mixed documents (emails, RFIs, change orders, invoices) from CPRA requests and discovery.

**Key User**: Non-technical attorneys needing intuitive UI and clear error messages.

## Quick Start
```bash
# Setup (first time only)
./scripts/setup_dev.sh

# Start backend
source venv/bin/activate
cd backend && uvicorn api.main:app --reload

# Start frontend (new terminal)
cd frontend && npm run dev

# Access
Frontend: http://localhost:5173
API Docs: http://localhost:8000/api/v1/docs
```

## Core Architecture

### Working Modules ✅
- **Document Processor**: PDF splitting with AI boundary detection (visual + pattern matching)
- **OCR System**: Adaptive preprocessing with 90%+ accuracy on CPRA documents
- **Storage**: SQLite + full-text search with FTS5
- **AI Classifier**: 80% accuracy using rule-based + DistilBERT ensemble
- **Metadata Extractor**: Dates, parties, amounts, reference numbers
- **Frontend**: Upload, list, view, delete documents

### Privacy Modes
- 🔒 **Full Local**: Everything on your computer
- 🔐 **Hybrid Safe**: Local processing, selective API use
- 🌐 **Full Featured**: Cloud APIs for enhanced features

## Key Files
```
backend/
├── modules/document_processor/
│   ├── pdf_splitter.py          # Main entry point
│   ├── hybrid_boundary_detector.py  # AI boundary detection
│   └── improved_ocr_handler.py  # OCR with caching
├── modules/ai_classifier/
│   └── classifier.py            # Document classification
├── modules/storage/
│   └── storage_manager.py       # Database operations
└── api/main.py                  # FastAPI app
```

## Current Development State

### What's Working
1. **Document Processing Pipeline**
   - Upload PDF → Split into documents → Classify → Store → View/Search
   - OCR: 90%+ confidence with disk caching (99.6% performance improvement)
   - Basic pipeline infrastructure and UI components

2. **Frontend Features**
   - Drag-drop upload with progress
   - Document list with sorting/filtering
   - PDF viewing and deletion
   - Privacy mode indicator

### Critical Issues Identified
1. **Accuracy Gap**: Current boundary detection and classification performs poorly on real-world documents
   - Test case: 36-page PDF with 13 documents → system detects only 2 boundaries, both misclassified
   - Synthetic test accuracy doesn't translate to real CPRA documents
2. **OCR-dependent Documents**: Pattern-matching fails on scanned/OCR'd content
3. **Complex Document Structures**: Visual similarity approach insufficient for legal document semantics

### **MAJOR UPGRADE IN PROGRESS**: LLM Integration
🚀 **Comprehensive LLM integration planned to dramatically improve accuracy**
- **Target**: >90% boundary detection, >85% classification (vs current ~20% real-world)
- **Approach**: Replace current classification entirely, enhance boundary detection with LLM validation
- **Models**: Llama 3 8B (local/Ollama) + GPT-4.1-nano (API when privacy allows)
- **Status**: Planning complete, implementation beginning
- **Details**: See `Llama-Integration.md` for comprehensive implementation plan

## Testing Critical Paths
```bash
# Test complete pipeline
python scripts/test_pipeline.py

# Test OCR system
python scripts/test_improved_ocr.py

# Test boundary detection
python scripts/test_visual_boundary.py

# Test classification
python scripts/test_ai_classifier.py
```

## Next Development Priorities

### **TOP PRIORITY: LLM Integration (4 weeks)**
**Objective**: Replace failing classification system with LLM-based approach for real-world accuracy

**Phase 1 (Week 1)**: Foundation Infrastructure
- Install Ollama + Llama 3 8B Instruct model
- Create LLM client abstraction (Ollama + OpenAI)
- Design and test classification/boundary detection prompts
- Establish performance baselines

**Phase 2 (Week 2)**: Classification Replacement  
- Implement LLM-based classifier with structured prompts
- Replace existing classifier.py entirely
- Test on problem documents (36-page PDF case)
- Measure accuracy improvements

**Phase 3 (Week 3)**: Boundary Detection Enhancement
- Add LLM boundary validation to hybrid detector
- Implement text windowing for context
- Integrate end-to-end pipeline
- Optimize for performance

**Phase 4 (Week 4)**: Production Readiness
- Error handling and fallback strategies
- Performance optimization and caching
- Monitoring and observability
- Update deployment documentation

### Deferred Until LLM Integration Complete
1. **UI Improvements**: Bulk operations, search interface, document detail views
2. **Advanced Features**: Graph engine, timeline builder, financial analyzer  
3. **Infrastructure**: Electron packaging, model fine-tuning

## Developer Notes

### Code Style
- Type hints EVERYWHERE
- Pydantic for validation
- Descriptive names: `extract_parties_from_document()` not `get_parties()`
- Error messages for attorneys, not developers

### Performance Targets
**Current**: Focus on accuracy over speed given real-world performance failures
- **Accuracy**: >90% boundary detection, >85% classification (primary goal)
- **Speed**: 5-10 seconds per document acceptable for LLM processing
- **Throughput**: 1000-page PDF processing in 15-30 minutes (vs current 5 min but wrong results)
- **Search**: <1 second response (when implemented)

### Common Workflows
```python
# Document processing pipeline
document = pdf_splitter.extract(pdf_path)
document = classifier.classify(document)
document = metadata_extractor.extract(document)
storage.save(document)

# Privacy-aware AI
if privacy_mode == PrivacyMode.FULL_LOCAL:
    result = local_model.process(document)
else:
    result = privacy_router.route(document, task)
```

### Gotchas
- OCR preprocessing can destroy clean PDFs - use adaptive strategies
- Boundary detection needs visual analysis for scanned docs
- FTS5 triggers have issues - populate manually
- Memory leaks with PyMuPDF pixmaps - always cleanup

## Useful Commands
```bash
# Check API health
curl http://localhost:8000/health

# Run specific tests
pytest backend/modules/document_processor/tests/ -v

# Format code
black backend/ && isort backend/

# Check types
mypy backend/

# View OCR cache stats
python -c "from backend.modules.document_processor.improved_ocr_handler import ImprovedOCRHandler; print(ImprovedOCRHandler().get_cache_stats())"
```

## References
- Project specs: `claim-specification.md`
- **LLM Integration Plan**: `Llama-Integration.md` (comprehensive implementation guide)
- API docs: http://localhost:8000/api/v1/docs
- Frontend: http://localhost:5173

---
**Remember**: This is for attorneys, not engineers. Prioritize clarity and reliability over technical elegance.