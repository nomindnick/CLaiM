# Document Browser Component

The Document Browser provides the main interface for attorneys to upload and manage construction litigation documents.

## Components

### DocumentUpload
- Drag-and-drop PDF upload interface using react-dropzone
- Multiple file upload support
- Real-time upload progress tracking
- File validation (PDF only, up to 100MB)
- Visual feedback for upload states

### DocumentBrowser
- Main container component with three view modes:
  - **Upload**: Document upload interface with guidelines
  - **List**: Table view of documents (to be implemented)
  - **Grid**: Card-based view of documents (to be implemented)

## Usage

```tsx
import { DocumentBrowser } from './components/document-browser'

// In your app
<DocumentBrowser />
```

## API Integration

The upload component sends files to:
- `POST /api/v1/documents/upload` - Multipart form data with PDF file

Expected response:
```json
{
  "document_id": "string",
  "status": "processing"
}
```

## Next Steps

1. Implement document list view with:
   - Sortable table columns
   - Search/filter functionality
   - Document type badges
   - Quick actions (view, download, delete)

2. Implement document grid view with:
   - Document preview thumbnails
   - Card-based layout
   - Hover effects

3. Add real-time processing status updates
4. Implement batch operations
5. Add document preview modal