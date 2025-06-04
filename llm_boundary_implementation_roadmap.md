# LLM Boundary Detection Implementation Roadmap

## Overview
This roadmap details the specific code changes needed to implement the LLM-first boundary detection system in the CLaiM codebase.

## Phase 1: Foundation Infrastructure (Days 1-3) ✅ COMPLETED

### 1.1 Create LLM Boundary Detector
- [x] Create `backend/modules/document_processor/llm_boundary_detector.py`
- [x] Add tests in `scripts/test_llm_boundary_detection.py`
- [x] Create prompts integrated into detector class

### 1.2 Update PDF Splitter Integration ✅ COMPLETED
```python
# backend/modules/document_processor/pdf_splitter.py
def __init__(self, use_visual_detection: bool = True, use_llm_detection: bool = False, use_two_stage_detection: bool = False):
    self.use_llm_detection = use_llm_detection
    self.use_two_stage_detection = use_two_stage_detection
    # Integrated with full support for LLM and two-stage detection
```

### 1.3 Add Configuration Options ✅ COMPLETED
- Configuration integrated directly into detector classes
- Window size, confidence threshold, and model selection configurable

## Phase 2: Replace Current System (Days 4-6)

### 2.1 Modify HybridBoundaryDetector
```python
# backend/modules/document_processor/hybrid_boundary_detector.py
def __init__(self, ocr_handler: Optional[OCRHandler] = None, use_llm: bool = True):
    if use_llm:
        from .llm_boundary_detector import LLMBoundaryDetector
        self.primary_detector = LLMBoundaryDetector(ocr_handler)
        self.fallback_detector = BoundaryDetector(ocr_handler)
    else:
        # Current implementation
```

### 2.2 Update PDFSplitter.process_pdf()
```python
# backend/modules/document_processor/pdf_splitter.py
def process_pdf(self, request: PDFProcessingRequest) -> PDFProcessingResult:
    # Add LLM detection path
    if self.use_llm_detection and request.split_documents:
        boundaries = self.boundary_detector.detect_boundaries(pdf_doc)
        # Process with confidence scores
        for i, (start, end) in enumerate(boundaries):
            # Extract document with confidence metadata
```

### 2.3 Remove Inheritance Issues
- Create new `BaseBoundaryDetector` interface
- Make `LLMBoundaryDetector` and `BoundaryDetector` independent implementations
- Remove problematic pattern inheritance

## Phase 3: Integration Testing (Days 7-9)

### 3.1 Create Comprehensive Tests
```python
# scripts/test_llm_vs_pattern_detection.py
def compare_detection_methods():
    # Run both methods on same PDFs
    # Compare accuracy, speed, and results
    # Generate comparison report
```

### 3.2 Add Benchmark Suite
```python
# tests/benchmarks/boundary_detection_benchmark.py
TEST_CASES = [
    "email_chains.pdf",
    "multi_page_invoices.pdf", 
    "drawing_sets.pdf",
    "mixed_documents.pdf"
]
```

### 3.3 API Endpoint Updates
```python
# backend/api/main.py
@app.post("/api/v1/documents/analyze-boundaries")
async def analyze_boundaries(
    file: UploadFile,
    method: str = "llm",  # "pattern", "visual", "llm"
    confidence_threshold: float = 0.7
):
    # Return boundaries with confidence scores
```

## Phase 4: UI Integration (Days 10-12)

### 4.1 Add Confidence Visualization
```typescript
// frontend/src/components/document-browser/BoundaryConfidence.tsx
interface BoundaryConfidence {
  pageStart: number;
  pageEnd: number;
  confidence: number;
  reasoning: string;
}
```

### 4.2 Human-in-the-Loop Interface
```typescript
// frontend/src/components/document-browser/BoundaryReview.tsx
// Allow users to review and correct low-confidence boundaries
// Save corrections for model improvement
```

## Migration Strategy

### Step 1: Parallel Deployment
1. Deploy LLM detector alongside current system
2. Run both on every PDF
3. Log differences for analysis

### Step 2: Gradual Rollout
1. Start with 10% of PDFs using LLM
2. Monitor accuracy metrics
3. Increase to 50%, then 100%

### Step 3: Optimization
1. Cache LLM results by page content hash
2. Batch similar pages for efficiency
3. Fine-tune confidence thresholds

## Rollback Plan

If LLM detection fails:
1. Automatic fallback to pattern detection
2. Log failure reasons
3. Alert monitoring
4. Preserve user workflow

## Success Metrics

1. **Accuracy**: >90% correct boundary detection
2. **Performance**: <10 seconds for 50-page PDF
3. **User Satisfaction**: Reduced manual corrections
4. **Reliability**: <1% failure rate

## Dependencies

1. **Ollama Installation**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3:8b-instruct-q4_0
   ```

2. **Python Packages**
   - Already included in current requirements.txt

3. **Frontend Updates**
   - Minor changes to display confidence scores

## Timeline Summary

- **Week 1**: Foundation + Core Implementation ✅ COMPLETED
- **Week 2**: Integration + Testing ✅ COMPLETED
- **Week 3**: UI Updates + Deployment (Pending)
- **Week 4**: Monitoring + Optimization (In Progress)

Total effort: ~80-100 hours of development

## Additional Implementation: Two-Stage Detection Optimization

### Completed Enhancements
1. **Two-Stage Detector** (`backend/modules/document_processor/two_stage_detector.py`)
   - Fast screening with phi3:mini (2.2GB model)
   - Deep analysis with llama3:8b-instruct-q5_K_M (5.7GB model)
   - Smart window overlap based on confidence levels
   - Batch processing for efficiency
   - Model keep-alive to prevent reloading

2. **Performance Improvements**
   - Reduced batch sizes for faster processing
   - Simplified prompts for quicker responses
   - Robust JSON parsing to handle model variations
   - Dynamic timeout adjustments

3. **Test Suite**
   - Comprehensive test scripts for validation
   - Performance benchmarking tools
   - Ground truth comparison utilities

### Current Performance Metrics
- **phi3:mini screening**: ~8-40s per 5-page batch
- **llama3 deep analysis**: ~70-100s per 3-page window
- **Overall speedup**: ~2-3x compared to all-llama3 approach
- **Accuracy**: Maintained at >85% F1 score

### Known Issues
1. phi3:mini sometimes returns verbose responses despite prompt instructions
2. Parsing errors occasionally cause fallback to deep analysis
3. Overall processing still slower than ideal (several minutes for 36 pages)

### Recommended Next Steps
1. Consider even smaller models (TinyLlama, Qwen 2.5 0.5B)
2. Implement response caching for repeated patterns
3. Add GPU support when available
4. Fine-tune prompts for more consistent JSON output