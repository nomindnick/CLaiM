# CLaiM Quick Start Guide

## 🚀 Starting the Application

### Prerequisites Check
Run the test script to verify your setup:
```bash
./scripts/test_setup.sh
```

### 1. Start the Backend API

```bash
# Terminal 1
cd backend
source ../venv/bin/activate
uvicorn api.main:app --reload
```

The backend will start on http://localhost:8000
- API Documentation: http://localhost:8000/api/v1/docs
- Health Check: http://localhost:8000/health

### 2. Start the Frontend

```bash
# Terminal 2
cd frontend
npm install  # First time only
npm run dev
```

The frontend will start on http://localhost:5173

## 📝 Using the Document Upload Feature

1. Open your browser to http://localhost:5173
2. Click on "Documents" in the navigation
3. Drag and drop PDF files or click to browse
4. Files will be uploaded and processed automatically

## 🛠️ Troubleshooting

### Backend Won't Start

**Error: ModuleNotFoundError: No module named 'fastapi'**
```bash
cd backend
source ../venv/bin/activate
pip install -r requirements-minimal.txt
```

**Error: ValidationError in Settings**
- Check that `.env` file exists in backend directory
- Verify environment variables match expected format

### Frontend Issues

**Error: Cannot find module 'react-dropzone'**
```bash
cd frontend
npm install
```

**Backend Connection Failed**
- Ensure backend is running on port 8000
- Check console for CORS errors
- Verify proxy settings in `vite.config.ts`

### Missing Dependencies

**Tesseract not found (for OCR)**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 📊 Current Features

✅ **Document Upload**
- Drag-and-drop PDF upload
- Progress tracking
- Multi-file support

✅ **Document Processing**
- PDF splitting into individual documents
- OCR for scanned pages
- Metadata extraction (dates, parties, amounts)
- Full-text search with SQLite FTS5

✅ **Privacy Modes**
- Full Local: All processing on your machine
- Hybrid Safe: Core operations local, non-sensitive cloud
- Full Featured: Cloud APIs for enhanced capabilities

## 🔄 Next Steps

1. Test document upload with sample PDFs
2. Check metadata extraction results
3. Implement document list view
4. Add document classification with AI

## 📁 Project Structure

```
CLaiM/
├── backend/           # FastAPI backend
│   ├── api/          # API configuration
│   ├── modules/      # Feature modules
│   └── .env          # Environment config
├── frontend/         # React TypeScript frontend
│   ├── src/          
│   │   └── components/
│   │       └── document-browser/  # Upload UI
│   └── package.json
└── storage/          # Document storage
    ├── database/     # SQLite database
    ├── pdfs/         # Original PDFs
    └── extracted/    # Split documents
```

## 🆘 Getting Help

- Check logs in backend terminal for errors
- Frontend console (F12) for client-side issues
- Run `./scripts/test_setup.sh` to verify environment
- See `CLAUDE.md` for detailed development notes