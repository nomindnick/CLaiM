# CLaiM - Construction Litigation AI Manager
## Specification & Development Guide

### Project Overview

#### Vision
CLaiM (Construction Litigation AI Manager) is a desktop application that helps attorneys manage and analyze large volumes of construction litigation documents, transforming unstructured PDFs into actionable intelligence for case preparation and claims analysis. The "ai" in CLaiM highlights its AI-powered capabilities while maintaining a privacy-first, local-processing approach.

#### Core Problem
Attorneys handling construction disputes receive massive PDF files (often 1000+ pages) containing mixed document types: emails, RFIs, change orders, invoices, contracts, and more. Current tools fail to:

- Identify individual documents within large PDFs
- Connect claims to supporting evidence
- Track financial transactions across documents
- Surface critical "smoking gun" documents
- Provide case-wide insights and timelines

#### Target Users
- **Primary**: Construction litigation attorneys
- **Secondary**: Paralegals and legal assistants
- **Environment**: Local desktop (Windows/Mac/Linux), no cloud dependency
- **Technical level**: Non-technical users requiring intuitive GUI

### Key User Stories

#### Document Management
1. As an attorney, I need to upload a large PDF and have it automatically split into individual logical documents so I can work with specific items rather than page numbers.
2. As an attorney, I need each document classified by type (RFI, email, invoice, etc.) and have metadata extracted (dates, parties, amounts) so I can quickly filter and find relevant documents.
3. As an attorney, I need to search using natural language ("show me all emails about the roof leak in June") and get relevant results with context.

#### Claims Analysis
1. As an attorney, I need to see how contractor claims connect to evidence so I can quickly assess their validity and prepare responses.
2. As an attorney, I need to identify documents that contradict specific claims so I can build effective defenses.
3. As an attorney, I need to trace the history of specific issues through multiple documents to understand the complete narrative.

#### Financial Tracking
1. As an attorney, I need to reconcile payment applications against actual payments to identify discrepancies.
2. As an attorney, I need to track change order requests through approval/rejection/payment to understand what remains disputed.

#### Case Intelligence
1. As an attorney, I need to automatically identify unusual or critical documents that might be "smoking guns" in the case.
2. As an attorney, I need to generate case chronologies showing key events and issues over time.
3. As an attorney, I need to export organized document sets with my analysis for discovery or case preparation.

### Technical Architecture

#### Core Stack
- **Backend**: Python 3.11+ with FastAPI
- **Frontend**: React 18+ with TypeScript
- **Desktop**: Electron (future phase)
- **Database**: SQLite (structured data) + Qdrant (vectors) + DuckDB (graph)
- **Classification Model**: DistilBERT (legal-domain fine-tuned)
- **Generation Model**: Phi-3.5-mini (3.8B, quantized)
- **Inference Engine**: llama.cpp for CPU optimization
- **Model Format**: GGUF for efficient deployment

#### Architecture Principles
- **Local-First**: All data and processing stays on user's machine
- **Privacy-Centric**: Three-tier privacy model for sensitive data
- **Modular**: Each feature is a self-contained module
- **Progressive**: Start with CLI/web, package as desktop app later
- **Offline-Capable**: Core features work without internet
- **AI-Augmented**: Use AI for enhancement, not dependency

#### System Architecture
```
User uploads PDF → Document Processor → Storage Layer → Analysis Modules → UI
                          ↓                    ↑              ↓
                   Classification          Search       Graph Builder
                     & Metadata           Engine      Timeline Builder
                                                    Financial Analyzer
```

### AI Model Architecture

#### Dual-Model System
1. **DistilBERT for Classification & Extraction**
   - Model: Legal-domain fine-tuned DistilBERT
   - Size: ~250MB (INT8 quantized: ~100MB)
   - Tasks: Document classification, NER, metadata extraction
   - Performance: <100ms per document on CPU

2. **Phi-3.5-mini for Generation**
   - Model: Microsoft Phi-3.5-mini (3.8B parameters)
   - Size: ~2GB (Q4_K_M quantized via GGUF)
   - Tasks: Summaries, narratives, complex analysis
   - Performance: 5-10 tokens/second on CPU
   - Context: 128K tokens (can process entire documents)

#### Privacy Modes
```python
class PrivacyMode(Enum):
    FULL_LOCAL = "No data leaves your computer"
    HYBRID_SAFE = "API calls only for non-sensitive analysis"
    FULL_FEATURED = "API calls for enhanced capabilities"
```

### Module Specifications

#### 1. Document Processor Module
**Purpose**: Transform raw PDFs into structured, searchable documents

**Key Functions**:
- PDF Splitting: Detect document boundaries using layout analysis and content patterns
- Document Classification: Categorize into ~15 types using DistilBERT
- OCR Handling: Process scanned pages to extract text
- Metadata Extraction: Pull dates, parties, reference numbers, amounts

**Technical Notes**:
- Use pymupdf for PDF manipulation (CPU-efficient)
- DistilBERT for fast classification (<100ms per doc)
- Smart boundary detection using headers/footers/signatures
- Store both extracted text and original page images

#### 2. Storage Module
**Purpose**: Persist and retrieve documents, metadata, and relationships

**Components**:
- SQLite Database: Document metadata, full-text search via FTS5
- Qdrant Vector Store: Document embeddings for semantic search
- DuckDB: Graph relationships between claims and evidence
- File System: Original PDFs, extracted pages, and GGUF models

**Key Design Decisions**:
- Document-oriented schema with JSON metadata fields
- Hierarchical storage: PDF → Document → Pages
- Immutable document store with versioned annotations
- Model weights stored in GGUF format for fast loading

#### 3. Search Module
**Purpose**: Enable flexible document discovery through multiple methods

**Search Types**:
- Keyword Search: SQLite FTS5 with legal term stemming
- Semantic Search: Vector similarity via Qdrant (using MiniLM embeddings)
- Structured Search: Filter by metadata (date ranges, parties, amounts)
- Hybrid Search: Combine all methods with weighted scoring

**Embedding Strategy**:
- Use all-MiniLM-L6-v2 for embeddings (90MB model)
- Pre-compute embeddings during document processing
- Cache embeddings in Qdrant for fast retrieval

#### 4. Graph Engine Module
**Purpose**: Build and query relationships between claims, facts, and evidence

**Graph Structure**:
- Nodes: Claims, Documents, Facts, Parties, Events
- Edges: Supports, Contradicts, References, Responds_To, Causes
- Properties: Confidence scores, dates, amounts

**Key Queries**:
- Evidence chain for specific claim
- Documents contradicting a fact
- Timeline of related events
- Missing evidence detection

#### 5. AI Interface Module
**Purpose**: Manage AI model interactions with intelligent routing

**Model Management**:
```python
class AIInterface:
    def __init__(self):
        # Always loaded for fast classification
        self.classifier = DistilBertClassifier("./models/distilbert-legal-Q8.bin")
        self.embedder = MiniLMEmbedder("./models/minilm-v2.bin")
        
        # Lazy loaded for generation tasks
        self.generator = None
        self.api_client = None
        
    def route_request(self, task, content, privacy_mode):
        if privacy_mode == PrivacyMode.FULL_LOCAL:
            return self._local_only_process(task, content)
        elif self._is_sensitive(content):
            return self._local_only_process(task, content)
        else:
            return self._hybrid_process(task, content)
```

**Smart Routing Logic**:
- Classification/extraction → Always local (DistilBERT)
- Simple summaries → Local (Phi-3.5-mini)
- Complex multi-doc analysis → API (if allowed)
- Sensitive content → Always local regardless of mode

#### 6. Financial Analyzer Module
**Purpose**: Track money flows and payment disputes

**Core Functions**:
- Table extraction from PDFs using local models
- Schedule of Values parsing with pattern matching
- Payment application tracking
- Change order reconciliation
- Disputed amount calculation

**Local Processing**:
- Use DistilBERT for identifying financial documents
- Rule-based extraction for standard formats
- Phi-3.5-mini for interpreting non-standard formats

#### 7. Timeline Builder Module
**Purpose**: Construct chronological narratives from documents

**Extraction Strategy**:
- DistilBERT extracts dates and events
- Graph engine builds relationships
- Phi-3.5-mini generates narrative summaries
- Critical path identification using graph algorithms

### Data Models

#### Document Model
```typescript
interface Document {
  id: string
  source_pdf_id: string
  type: DocumentType  // 'RFI' | 'Email' | 'Invoice' | etc.
  pages: number[]     // Page numbers in source PDF
  
  // Extracted metadata
  date: Date
  parties: Party[]
  title: string
  reference_numbers: string[]  // RFI#, CO#, Invoice#
  amounts: Amount[]
  
  // Content
  text: string
  tables: Table[]
  
  // AI-generated
  embedding: number[]  // MiniLM embeddings
  summary: string
  key_facts: Fact[]
  classification_confidence: number
  
  // Relationships
  responds_to?: string[]  // Document IDs
  references?: string[]
}
```

#### Claim-Evidence Graph
```typescript
interface ClaimNode {
  id: string
  description: string
  claimed_amount?: number
  category: 'delay' | 'change' | 'defect' | 'payment'
  status: 'disputed' | 'accepted' | 'rejected'
  supporting_evidence: EvidenceEdge[]
  contradicting_evidence: EvidenceEdge[]
}

interface EvidenceEdge {
  from: string  // Document or Fact ID
  to: string    // Claim ID
  relationship: 'supports' | 'contradicts' | 'mentions'
  confidence: number  // 0-1
  explanation?: string  // Generated by Phi-3.5
}
```

### UI/UX Design Principles

#### Layout Philosophy
- **Information Density**: Construction attorneys work with details; don't oversimplify
- **Multi-Panel Interface**: Support comparing multiple documents simultaneously
- **Persistent Context**: Keep search/filter state while navigating
- **Privacy Indicators**: Clear visual feedback on data processing location

#### Core Views
1. **Document Browser**: Primary interface, similar to email client
2. **Graph Explorer**: Interactive visualization of claim-evidence relationships
3. **Timeline View**: Chronological event display with swim lanes
4. **Claims Workbench**: Focused interface for analyzing specific claims
5. **Financial Dashboard**: Payment tracking and reconciliation

#### Privacy UI Elements
- 🔒 Green lock: Full local processing
- 🔐 Yellow shield: Hybrid mode (selective API use)
- 🌐 Blue globe: Full API features enabled
- Clear indicators when switching modes
- Confirmation dialogs for mode changes

### Performance Specifications

#### Hardware Requirements
**Minimum**:
- CPU: 4-core x86/ARM processor
- RAM: 8GB
- Storage: 10GB free space

**Recommended**:
- CPU: 8-core processor (e.g., AMD Ryzen 7840U)
- RAM: 16-32GB
- Storage: 50GB free space

#### Performance Targets
**Document Processing**:
- Classification: <100ms per document
- Full processing: <5 seconds per document
- 1000-page PDF: <5 minutes total

**Search & Retrieval**:
- Keyword search: <100ms
- Semantic search: <500ms
- Combined search: <1 second

**AI Generation** (Local):
- Summary generation: 5-10 tokens/second
- First token latency: <500ms
- Context window: 128K tokens

**Memory Usage**:
- Base application: ~500MB
- With models loaded: ~4GB
- Peak (processing): ~8GB

### Implementation Phases

#### Phase 1: Foundation (MVP)
- Document processor with DistilBERT classification
- SQLite storage with simple search
- Basic web UI with document browser
- Local-only mode implementation

#### Phase 2: Intelligence
- Phi-3.5-mini integration for generation
- Vector search via Qdrant
- Graph engine basics
- Privacy mode selection

#### Phase 3: Analysis
- Financial analyzer
- Timeline builder
- Claims workbench
- API integration for enhanced features

#### Phase 4: Production
- Electron packaging
- Model fine-tuning pipeline
- Performance optimization
- Multi-user support

### Model Management

#### Deployment Strategy
```
claim/
├── models/
│   ├── distilbert-legal.gguf      # 250MB - Always loaded
│   ├── phi-3.5-mini-Q4_K_M.gguf  # 2GB - Lazy loaded
│   └── minilm-embedder.gguf      # 90MB - Always loaded
```

#### Fine-tuning Pipeline
1. Collect firm-specific training data (with consent)
2. Quarterly fine-tuning cycles
3. Cloud-based training (Google Colab Pro recommended)
4. Local validation before deployment
5. Distribute updates via application updater

### Security & Privacy

#### Data Protection
- All documents encrypted at rest
- No telemetry without explicit consent
- Audit trail for all operations
- Secure API key storage (when used)

#### Privacy Guarantees
- Full local mode: Zero external communication
- Hybrid mode: Only non-sensitive data to APIs
- Clear data flow visualization
- User controls over each AI operation

### Success Metrics
- Process 1000-page PDF in <5 minutes
- Document classification accuracy >95%
- Search returns relevant results in <1 second
- Can trace claim to evidence in <3 clicks
- Financial reconciliation matches manual review
- Runs smoothly on standard attorney laptops
- <3GB total installation size

### Resources and References
- Construction document types: RFI, RFP, CO, ASI, Submittal, Pay App
- Legal search requirements: Boolean operators, proximity search, stemming
- Model references: DistilBERT, Phi-3.5-mini, llama.cpp, GGUF format
- UI frameworks: React, Electron, D3.js for visualization