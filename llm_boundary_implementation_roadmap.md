# LLM Boundary Detection Implementation Roadmap

## Overview
This roadmap details the specific code changes needed to implement the LLM-first boundary detection system in the CLaiM codebase.

## Phase 1: Foundation Infrastructure (Days 1-3)

### 1.1 Create LLM Boundary Detector
- [x] Create `backend/modules/document_processor/llm_boundary_detector.py`
- [ ] Add tests in `backend/modules/document_processor/tests/test_llm_boundary_detector.py`
- [ ] Create prompts in `backend/modules/llm_client/prompts/boundary_detection.py`

### 1.2 Update PDF Splitter Integration
```python
# backend/modules/document_processor/pdf_splitter.py
def __init__(self, use_visual_detection: bool = True, use_llm_detection: bool = False):
    self.use_llm_detection = use_llm_detection
    if use_llm_detection:
        from .llm_boundary_detector import LLMBoundaryDetector
        self.boundary_detector = LLMBoundaryDetector()
```

### 1.3 Add Configuration Options
```python
# backend/api/config.py
BOUNDARY_DETECTION_MODE = os.getenv("BOUNDARY_DETECTION_MODE", "llm")  # "pattern", "visual", "llm"
LLM_WINDOW_SIZE = int(os.getenv("LLM_WINDOW_SIZE", "3"))
LLM_CONFIDENCE_THRESHOLD = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.7"))
```

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

- **Week 1**: Foundation + Core Implementation
- **Week 2**: Integration + Testing
- **Week 3**: UI Updates + Deployment
- **Week 4**: Monitoring + Optimization

Total effort: ~80-100 hours of development