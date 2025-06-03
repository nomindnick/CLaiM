# LLM-First Document Intelligence Architecture

## Core Principles

1. **Semantic Understanding Over Pattern Matching**: The LLM understands document context, not just keywords
2. **Sliding Window Context**: Analyze documents with overlapping windows for continuity detection  
3. **Confidence-Based Decisions**: Probabilistic boundaries with threshold tuning
4. **Document-Type Aware**: Different strategies for emails vs invoices vs drawings
5. **Iterative Refinement**: Multiple passes with increasing context when needed

## Architecture Components

### 1. Document Pre-Processor
```python
class DocumentPreProcessor:
    """Prepares PDF for LLM analysis."""
    
    def extract_page_windows(self, pdf_doc, window_size=3, overlap=1):
        """Extract overlapping text windows for boundary analysis."""
        windows = []
        for i in range(0, pdf_doc.page_count, window_size - overlap):
            window_pages = []
            for j in range(i, min(i + window_size, pdf_doc.page_count)):
                text = self._get_page_text(pdf_doc[j])
                window_pages.append({
                    'page_num': j,
                    'text': text,
                    'has_images': len(pdf_doc[j].get_images()) > 0,
                    'has_drawings': len(pdf_doc[j].get_drawings()) > 10
                })
            windows.append(window_pages)
        return windows
```

### 2. LLM Boundary Analyzer
```python
class LLMBoundaryAnalyzer:
    """Uses LLM to analyze document boundaries with confidence scores."""
    
    def analyze_window(self, window_pages, previous_context=None):
        """Analyze a window of pages for document boundaries."""
        
        prompt = self._build_analysis_prompt(window_pages, previous_context)
        
        # LLM returns structured analysis
        analysis = self.llm_client.analyze(prompt, response_format={
            "boundaries": [
                {
                    "page_num": "int",
                    "confidence": "float (0-1)",
                    "document_type": "string",
                    "reasoning": "string",
                    "continues_from_previous": "boolean"
                }
            ],
            "window_summary": "string"
        })
        
        return analysis
```

### 3. Boundary Consolidator
```python
class BoundaryConsolidator:
    """Consolidates overlapping window analyses into final boundaries."""
    
    def consolidate_boundaries(self, window_analyses, confidence_threshold=0.7):
        """Merge window analyses into final document boundaries."""
        
        # Aggregate confidence scores for each potential boundary
        boundary_votes = defaultdict(list)
        
        for analysis in window_analyses:
            for boundary in analysis['boundaries']:
                boundary_votes[boundary['page_num']].append({
                    'confidence': boundary['confidence'],
                    'type': boundary['document_type'],
                    'reasoning': boundary['reasoning']
                })
        
        # Determine final boundaries based on consensus
        final_boundaries = []
        for page_num, votes in boundary_votes.items():
            avg_confidence = sum(v['confidence'] for v in votes) / len(votes)
            if avg_confidence >= confidence_threshold:
                # Use majority vote for document type
                type_votes = Counter(v['type'] for v in votes)
                document_type = type_votes.most_common(1)[0][0]
                
                final_boundaries.append({
                    'page_num': page_num,
                    'confidence': avg_confidence,
                    'document_type': document_type,
                    'supporting_analyses': len(votes)
                })
        
        return final_boundaries
```

### 4. Document Type Strategies
```python
class DocumentTypeStrategy:
    """Document-type specific boundary detection strategies."""
    
    @staticmethod
    def get_strategy(document_type):
        strategies = {
            'email': EmailChainStrategy(),
            'invoice': InvoiceStrategy(),
            'drawing': TechnicalDrawingStrategy(),
            'rfi': RFIStrategy(),
            'submittal': SubmittalStrategy()
        }
        return strategies.get(document_type, DefaultStrategy())

class EmailChainStrategy:
    def validate_continuation(self, current_page, previous_page):
        """Check if current page continues an email chain."""
        # LLM prompt specifically for email continuity
        prompt = f"""
        Analyze if these two pages are part of the same email chain:
        
        Previous page ending:
        {previous_page['text'][-500:]}
        
        Current page beginning:
        {current_page['text'][:500]}
        
        Consider:
        1. Is this a quoted reply (>, >>, etc.)?
        2. Does the subject line match?
        3. Is this a forwarded section?
        4. Are the participants the same?
        
        Return: {{"continues": true/false, "confidence": 0.0-1.0, "reason": "..."}}
        """
        return self.llm_client.analyze(prompt)
```

### 5. Prompt Engineering
```python
BOUNDARY_ANALYSIS_PROMPT = """
You are analyzing pages from a construction litigation PDF to identify document boundaries.

Window of pages to analyze:
{window_content}

Previous window summary (if any):
{previous_summary}

For each page in this window, determine:
1. Is this the start of a new document? (confidence: 0.0-1.0)
2. What type of document is it? (email, rfi, invoice, submittal, drawing, etc.)
3. Does it continue from the previous window?

Consider these patterns:
- Emails: Look for From/To/Subject headers, but also consider forwarded/quoted sections
- Multi-page documents: Invoices, RFIs, and drawings often span multiple pages
- Continuation indicators: "Page X of Y", consistent headers/footers, continued item lists
- Visual changes: Significant format changes may indicate new documents

Return a structured analysis for each page.
"""
```

## Implementation Plan

### Phase 1: Core LLM Infrastructure (Week 1)
1. Implement `DocumentPreProcessor` with sliding window extraction
2. Create `LLMBoundaryAnalyzer` with structured prompt templates
3. Build `BoundaryConsolidator` for merging window analyses
4. Add confidence threshold tuning

### Phase 2: Document Type Strategies (Week 2)
1. Implement strategy pattern for document types
2. Create specialized prompts for each document type
3. Add multi-page document understanding
4. Build continuity detection logic

### Phase 3: Classification Integration (Week 3)
1. Replace current classifier with LLM-based approach
2. Use boundary detection context for better classification
3. Implement metadata extraction in same pass
4. Add confidence scores to classifications

### Phase 4: Optimization & Validation (Week 4)
1. Implement caching for LLM calls
2. Add batch processing for efficiency
3. Create feedback loop for low-confidence boundaries
4. Extensive testing on real documents

## Key Advantages

1. **Semantic Understanding**: LLM understands "this is a forwarded email" not just "contains 'From:'"
2. **Context Awareness**: Sliding windows provide document continuity context
3. **Confidence Scoring**: Can flag uncertain boundaries for human review
4. **Document-Type Intelligence**: Knows that drawings come in sets, emails have threads
5. **Iterative Improvement**: Can add more context for difficult decisions

## Migration Strategy

1. Keep existing system as fallback
2. Implement new system in parallel
3. A/B test on real documents
4. Gradually increase confidence threshold
5. Monitor accuracy metrics

## Expected Outcomes

- **Boundary Detection**: >90% accuracy (vs current 28-50%)
- **Classification**: >85% accuracy 
- **Processing Time**: 5-10 seconds per document (acceptable)
- **Confidence Scores**: Enable human-in-the-loop for edge cases