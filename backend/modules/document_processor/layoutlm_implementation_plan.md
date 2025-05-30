# LayoutLM Boundary Detection - Full Implementation Plan

## Current Status

The LayoutLM boundary detector is partially implemented but has several critical issues:

1. **Model Not Fine-tuned**: Using base LayoutLMv3 without training for boundary detection
2. **No Training Data**: Need labeled construction document boundaries
3. **Incorrect Logic**: Current implementation uses untrained classification head
4. **Configuration Issues**: Processor misconfigured for our use case

## What's Needed for Full Implementation

### 1. Data Collection & Annotation
```python
# Need to create training dataset with:
training_data = [
    {
        "pdf_path": "construction_doc_001.pdf",
        "boundaries": [(0, 5), (6, 12), (13, 18)],  # Ground truth
        "page_labels": [1, 0, 0, 0, 0, 1, 0, ...]  # 1 = boundary, 0 = continuation
    },
    # ... hundreds more examples
]
```

### 2. Model Architecture Changes
Instead of sequence classification, we need token classification or custom approach:

```python
class BoundaryDetectionModel(nn.Module):
    def __init__(self, layoutlm_model):
        super().__init__()
        self.layoutlm = layoutlm_model
        self.boundary_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # boundary/no-boundary
        )
        
    def forward(self, **inputs):
        outputs = self.layoutlm(**inputs)
        # Use CLS token or aggregate page features
        page_representation = outputs.last_hidden_state[:, 0, :]
        boundary_logits = self.boundary_classifier(page_representation)
        return boundary_logits
```

### 3. Training Pipeline
```python
def train_boundary_detector(train_dataset, val_dataset):
    # 1. Prepare data loaders
    # 2. Initialize model with pre-trained LayoutLMv3
    # 3. Fine-tune on construction documents
    # 4. Focus on:
    #    - Document headers/letterheads
    #    - Signature blocks
    #    - Form transitions
    #    - Date changes
    #    - Reference number patterns
```

### 4. Inference Improvements
Current implementation processes pages individually. Better approach:

```python
def detect_boundaries_windowed(self, pdf_doc, window_size=3):
    """Process pages in sliding windows for context."""
    boundaries = []
    
    for i in range(1, pdf_doc.page_count):
        # Look at pages [i-1, i, i+1] together
        window_start = max(0, i - window_size // 2)
        window_end = min(pdf_doc.page_count, i + window_size // 2 + 1)
        
        # Extract features from window
        window_features = self._extract_window_features(
            pdf_doc, window_start, window_end, target_page=i
        )
        
        # Predict if page i is a boundary
        is_boundary = self._predict_boundary(window_features)
        
        if is_boundary:
            boundaries.append(i)
    
    return boundaries
```

### 5. Construction-Specific Features
Add domain knowledge:

```python
def extract_construction_features(self, page):
    """Extract construction-specific features."""
    features = {
        "has_letterhead": self._detect_letterhead(page),
        "has_stamps": self._detect_stamps(page),  # Architect/engineer stamps
        "has_signature_block": self._detect_signatures(page),
        "document_type_keywords": self._extract_doc_type_keywords(page),
        "form_structure": self._analyze_form_layout(page),
        "drawing_elements": self._detect_technical_drawings(page),
    }
    return features
```

## Implementation Steps

### Phase 1: Data Preparation (1-2 weeks)
1. Collect 500+ construction PDFs with various document types
2. Manually annotate boundaries
3. Create train/validation/test splits
4. Build data loading pipeline

### Phase 2: Model Development (2-3 weeks)
1. Design custom architecture for boundary detection
2. Implement training loop with proper metrics
3. Experiment with different approaches:
   - Token classification
   - Sequence-to-sequence
   - Pairwise page comparison
4. Add construction-specific features

### Phase 3: Training & Evaluation (1 week)
1. Train on GPU cluster
2. Hyperparameter tuning
3. Evaluate on test set
4. Compare with visual detection baseline

### Phase 4: Integration (1 week)
1. Optimize inference speed
2. Add confidence calibration
3. Integrate with hybrid detector
4. Add explainability features

## Alternative: Simpler Heuristic Approach

If full training isn't feasible, we could improve the current implementation:

```python
def detect_boundaries_heuristic(self, pdf_doc):
    """Use LayoutLM embeddings with heuristic rules."""
    embeddings = []
    
    # Extract embeddings for each page
    for page in pdf_doc:
        embedding = self._get_page_embedding(page)
        embeddings.append(embedding)
    
    boundaries = []
    for i in range(1, len(embeddings)):
        # Compare consecutive pages
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        
        # Heuristic: boundary if similarity < threshold
        # AND page has certain layout features
        if similarity < 0.7 and self._has_boundary_features(pdf_doc[i]):
            boundaries.append(i)
    
    return boundaries
```

## Recommendation

For immediate use, I recommend:

1. **Stick with Visual Detection**: It's working well and doesn't require training
2. **Collect Training Data**: Start gathering labeled construction documents
3. **Prototype Heuristic Approach**: Use LayoutLM embeddings with rules
4. **Plan for Full Implementation**: Budget time/resources for proper training

The visual detection (CLIP) is already providing good results. LayoutLM would be most valuable for:
- Complex multi-column layouts
- Forms with similar visual appearance but different content
- Documents where text position/structure matters more than visual appearance