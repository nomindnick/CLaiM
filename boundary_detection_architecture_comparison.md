# Boundary Detection Architecture Comparison

## Current Pattern-Based Architecture

```
PDF Document (36 pages)
    ↓
[Page-by-Page Analysis]
    ↓
For each page:
  - Extract text
  - Match against patterns:
    * Contains "From:" → NEW DOCUMENT!
    * Contains "Subject:" → NEW DOCUMENT!
    * Contains "PACKING SLIP" → NEW DOCUMENT!
    ↓
[Binary Decision: Is Boundary?]
    ↓
Result: 25 documents (should be 14)
```

### Problems Illustrated
- Page 2: "From: John" in quoted email → Incorrectly marked as new document
- Page 18-22: Each packing slip page → Split into 5 documents (should be 1)
- Page 27-31: Drawing sheets 1-5 → Split into 5 documents (should be 1)

## New LLM-Based Architecture

```
PDF Document (36 pages)
    ↓
[Sliding Window Extraction]
    ↓
Windows: [1,2,3], [3,4,5], [5,6,7]...
    ↓
For each window:
  ┌─────────────────────────────┐
  │         LLM Analysis        │
  │ "These 3 pages contain..."  │
  │ "Page 2 appears to be..."   │
  │ "This continues from..."    │
  └─────────────────────────────┘
    ↓
[Confidence Scores per Boundary]
  - Page 1: 0.95 (Email start)
  - Page 2: 0.15 (Quoted content)
  - Page 5: 0.92 (New email)
    ↓
[Voting & Consolidation]
    ↓
Result: 14 documents ✓
```

### Solutions Illustrated
- Page 2: LLM recognizes quoted email context → Correctly continues document
- Page 18-22: LLM sees same vendor/order → Keeps as single document
- Page 27-31: LLM recognizes drawing sequence → Groups as drawing set

## Key Architectural Differences

### 1. Analysis Unit
- **Current**: Single page in isolation
- **New**: 3-page sliding windows with context

### 2. Decision Making
- **Current**: Binary pattern matching
- **New**: Probabilistic confidence scoring

### 3. Understanding
- **Current**: Surface-level text patterns
- **New**: Semantic document understanding

### 4. Continuity Detection
- **Current**: None (each page independent)
- **New**: Explicit continuity analysis

### 5. Adaptability
- **Current**: Hard-coded patterns
- **New**: Prompt-based configuration

## Example: Email Thread Detection

### Current System
```python
# Sees "From:" on any page
if re.search(r"From:", text):
    return True  # NEW DOCUMENT!
```

### LLM System
```
LLM Analysis of Pages 1-3:
"Page 1 starts a new email thread about RFI #45.
Page 2 contains quoted replies with '>' markers.
Page 3 continues the email discussion.
These pages form a single email conversation."

Boundaries:
- Page 1: confidence=0.95 (new email thread)
- Page 2: confidence=0.10 (continuation)
- Page 3: confidence=0.05 (continuation)
```

## Performance Trade-offs

| Metric | Current System | LLM System |
|--------|---------------|------------|
| Speed | <1 second | 5-10 seconds |
| Accuracy | 28-50% | >90% expected |
| Context | None | 3-page windows |
| Explanation | None | Full reasoning |
| Maintenance | Code changes | Prompt updates |

## Visual Example: Processing Flow

### Current: Linear, Isolated
```
Page 1 → Pattern Match → Boundary? → Next Page
Page 2 → Pattern Match → Boundary? → Next Page
Page 3 → Pattern Match → Boundary? → Next Page
```

### New: Windowed, Contextual
```
[Pages 1-3] → LLM → "Email thread about RFI"
    ↓ (overlap)
[Pages 3-5] → LLM → "Email continues, new email starts at 5"
    ↓ (overlap)
[Pages 5-7] → LLM → "New email about invoice"
```

## Conclusion

The fundamental shift from pattern matching to semantic understanding addresses the root cause of the current system's failures. While slightly slower, the LLM approach provides the accuracy and intelligence needed for reliable document processing in the CLaiM application.