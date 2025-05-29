# CLaiM - Project Structure

```
claim/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app entry point
│   │   ├── config.py                   # Global configuration
│   │   └── privacy_manager.py          # Privacy mode handling
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   │
│   │   ├── document_processor/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Pydantic models for documents
│   │   │   ├── pdf_splitter.py         # PDF to logical documents
│   │   │   ├── boundary_detector.py    # Document boundary detection
│   │   │   ├── ocr_handler.py          # OCR for scanned pages
│   │   │   ├── router.py               # FastAPI routes
│   │   │   └── tests/
│   │   │       ├── test_splitter.py
│   │   │       └── test_boundary.py
│   │   │
│   │   ├── ai_classifier/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Classification models
│   │   │   ├── distilbert_classifier.py # DistilBERT implementation
│   │   │   ├── document_types.py       # Document type definitions
│   │   │   ├── confidence_scorer.py    # Classification confidence
│   │   │   ├── router.py
│   │   │   └── tests/
│   │   │
│   │   ├── metadata_extractor/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Metadata schemas
│   │   │   ├── ner_extractor.py        # Named entity recognition
│   │   │   ├── date_extractor.py       # Extract dates
│   │   │   ├── amount_extractor.py     # Extract money amounts
│   │   │   ├── party_extractor.py      # Extract parties/entities
│   │   │   ├── reference_extractor.py  # Extract doc references
│   │   │   ├── router.py
│   │   │   └── tests/
│   │   │
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Database models
│   │   │   ├── sqlite_handler.py       # SQLite + FTS5 operations
│   │   │   ├── vector_store.py         # Qdrant operations
│   │   │   ├── graph_store.py          # DuckDB graph operations
│   │   │   ├── file_store.py           # File system operations
│   │   │   ├── model_store.py          # GGUF model management
│   │   │   └── tests/
│   │   │
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Search query/result models
│   │   │   ├── keyword_search.py       # FTS5 keyword search
│   │   │   ├── semantic_search.py      # Vector similarity search
│   │   │   ├── hybrid_search.py        # Combined search strategies
│   │   │   ├── query_parser.py         # Natural language parsing
│   │   │   ├── embedder.py             # MiniLM embeddings
│   │   │   ├── reranker.py             # Result reranking
│   │   │   ├── router.py
│   │   │   └── tests/
│   │   │
│   │   ├── graph_engine/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Graph node/edge models
│   │   │   ├── graph_builder.py        # Build claim-evidence graph
│   │   │   ├── graph_queries.py        # Path finding, analytics
│   │   │   ├── relationship_scorer.py  # Score claim-evidence links
│   │   │   ├── router.py
│   │   │   └── tests/
│   │   │
│   │   ├── ai_interface/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # AI request/response models
│   │   │   ├── model_manager.py        # Load/unload models
│   │   │   ├── local_classifier.py     # DistilBERT interface
│   │   │   ├── local_generator.py      # Phi-3.5 interface
│   │   │   ├── api_client.py           # OpenAI/Claude interface
│   │   │   ├── privacy_router.py       # Route based on privacy mode
│   │   │   ├── prompt_templates.py     # Task-specific prompts
│   │   │   ├── llama_cpp_wrapper.py    # llama.cpp integration
│   │   │   └── tests/
│   │   │
│   │   ├── financial_analyzer/
│   │   │   ├── __init__.py
│   │   │   ├── models.py               # Financial data models
│   │   │   ├── table_extractor.py      # Extract tables from PDFs
│   │   │   ├── sov_parser.py           # Schedule of Values parser
│   │   │   ├── payment_tracker.py      # Track payments/changes
│   │   │   ├── amount_reconciler.py    # Reconcile financials
│   │   │   ├── router.py
│   │   │   └── tests/
│   │   │
│   │   └── timeline_builder/
│   │       ├── __init__.py
│   │       ├── models.py               # Timeline event models
│   │       ├── event_extractor.py      # Extract timeline events
│   │       ├── chronology.py           # Build chronology
│   │       ├── narrative_generator.py  # Generate timeline narratives
│   │       ├── router.py
│   │       └── tests/
│   │
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── exceptions.py               # Custom exceptions
│   │   ├── utils.py                    # Shared utilities
│   │   ├── constants.py                # Shared constants
│   │   ├── privacy.py                  # Privacy mode definitions
│   │   └── performance.py              # Performance monitoring
│   │
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── pytest.ini
│   └── .env.example
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── claim-logo.svg              # CLaiM logo
│   │
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.tsx
│   │   │
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   ├── LoadingSpinner.tsx
│   │   │   │   ├── PrivacyIndicator.tsx  # Show current privacy mode
│   │   │   │   └── ModelStatus.tsx        # Show loaded models
│   │   │   │
│   │   │   ├── document-browser/
│   │   │   │   ├── DocumentBrowser.tsx
│   │   │   │   ├── DocumentList.tsx
│   │   │   │   ├── DocumentViewer.tsx
│   │   │   │   ├── DocumentFilters.tsx
│   │   │   │   ├── ClassificationBadge.tsx
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   ├── graph-explorer/
│   │   │   │   ├── GraphExplorer.tsx
│   │   │   │   ├── GraphCanvas.tsx
│   │   │   │   ├── NodeDetails.tsx
│   │   │   │   ├── GraphControls.tsx
│   │   │   │   ├── ClaimEvidenceView.tsx
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   ├── timeline-view/
│   │   │   │   ├── TimelineView.tsx
│   │   │   │   ├── TimelineEvent.tsx
│   │   │   │   ├── ChronologyBuilder.tsx
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   ├── query-interface/
│   │   │   │   ├── QueryInterface.tsx
│   │   │   │   ├── QueryInput.tsx
│   │   │   │   ├── QueryResults.tsx
│   │   │   │   ├── SearchFilters.tsx
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   ├── claims-workbench/
│   │   │   │   ├── ClaimsWorkbench.tsx
│   │   │   │   ├── ClaimsList.tsx
│   │   │   │   ├── EvidenceMapper.tsx
│   │   │   │   ├── ClaimAnalysis.tsx
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   ├── financial-dashboard/
│   │   │   │   ├── FinancialDashboard.tsx
│   │   │   │   ├── PaymentWaterfall.tsx
│   │   │   │   ├── ChangeOrderGrid.tsx
│   │   │   │   ├── ReconciliationView.tsx
│   │   │   │   └── types.ts
│   │   │   │
│   │   │   └── settings/
│   │   │       ├── SettingsPanel.tsx
│   │   │       ├── PrivacySettings.tsx
│   │   │       ├── ModelSettings.tsx
│   │   │       └── APIKeyManager.tsx
│   │   │
│   │   ├── services/
│   │   │   ├── api.ts                  # API client
│   │   │   ├── documentService.ts
│   │   │   ├── searchService.ts
│   │   │   ├── graphService.ts
│   │   │   ├── aiService.ts            # AI model interactions
│   │   │   ├── privacyService.ts       # Privacy mode management
│   │   │   └── types.ts
│   │   │
│   │   ├── hooks/
│   │   │   ├── useDocuments.ts
│   │   │   ├── useSearch.ts
│   │   │   ├── useGraph.ts
│   │   │   ├── useAI.ts                # AI model status/calls
│   │   │   └── usePrivacy.ts           # Privacy mode hook
│   │   │
│   │   ├── store/
│   │   │   ├── index.ts                # Redux/Zustand store
│   │   │   ├── documentSlice.ts
│   │   │   ├── uiSlice.ts
│   │   │   ├── aiSlice.ts              # AI model state
│   │   │   └── privacySlice.ts         # Privacy settings
│   │   │
│   │   └── utils/
│   │       ├── formatters.ts
│   │       ├── validators.ts
│   │       └── constants.ts
│   │
│   ├── package.json
│   ├── tsconfig.json
│   ├── .eslintrc.js
│   └── vite.config.ts
│
├── models/                              # Pre-trained models
│   ├── distilbert-legal/
│   │   ├── config.json
│   │   ├── model.gguf                  # Quantized DistilBERT
│   │   └── tokenizer.json
│   │
│   ├── phi-3.5-mini/
│   │   ├── phi-3.5-mini-Q4_K_M.gguf   # 4-bit quantized
│   │   └── config.json
│   │
│   ├── embeddings/
│   │   ├── all-MiniLM-L6-v2.gguf      # Sentence embeddings
│   │   └── config.json
│   │
│   └── download_models.py              # Model download script
│
├── electron/                            # Desktop packaging
│   ├── main.js
│   ├── preload.js
│   ├── package.json
│   └── resources/
│       ├── icon.ico
│       └── installer/
│
├── scripts/
│   ├── setup_dev.sh                    # Development environment setup
│   ├── download_models.sh              # Download required models
│   ├── convert_to_gguf.py              # Convert models to GGUF
│   ├── build_dist.sh                   # Build distribution packages
│   ├── fine_tune_classifier.py         # Fine-tune DistilBERT
│   └── sample_data_generator.py        # Generate test data
│
├── training/                            # Model fine-tuning
│   ├── data/
│   │   ├── classification/             # Document type training data
│   │   └── ner/                        # NER training data
│   │
│   ├── notebooks/
│   │   ├── distilbert_finetuning.ipynb
│   │   └── evaluation_metrics.ipynb
│   │
│   └── configs/
│       ├── distilbert_config.yaml
│       └── training_config.yaml
│
├── docs/
│   ├── API.md                          # API documentation
│   ├── MODULES.md                      # Module descriptions
│   ├── AI_ARCHITECTURE.md              # AI system design
│   ├── PRIVACY_GUIDE.md                # Privacy mode documentation
│   ├── DEVELOPMENT.md                  # Development guide
│   ├── DEPLOYMENT.md                   # Deployment guide
│   └── FINE_TUNING.md                  # Model fine-tuning guide
│
├── docker/                              # Optional containerization
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── tests/
│   ├── integration/                     # Integration tests
│   ├── e2e/                            # End-to-end tests
│   └── benchmarks/                     # Performance benchmarks
│       ├── classification_speed.py
│       ├── generation_speed.py
│       └── memory_usage.py
│
├── .gitignore
├── .gitattributes                      # LFS settings for models
├── README.md
├── LICENSE
├── CLAUDE.md                           # Claude Code integration
└── pyproject.toml                      # Python project config
```