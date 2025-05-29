import React, { useState } from 'react'
import { DocumentUpload } from './DocumentUpload'
import { FolderOpen, Upload, Grid, List } from 'lucide-react'

type ViewMode = 'upload' | 'list' | 'grid'

export const DocumentBrowser: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('upload')
  const [uploadedDocuments, setUploadedDocuments] = useState<string[]>([])

  const handleUploadComplete = (documentId: string) => {
    setUploadedDocuments(prev => [...prev, documentId])
    // In a real app, you'd refetch the document list here
    console.log('Document uploaded:', documentId)
  }

  const handleUploadError = (error: string) => {
    console.error('Upload error:', error)
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b bg-white px-6 py-4">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Document Browser</h2>
            <p className="text-sm text-gray-600 mt-1">
              Upload and manage construction litigation documents
            </p>
          </div>

          {/* View Mode Switcher */}
          <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('upload')}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                transition-colors duration-200
                ${viewMode === 'upload'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <Upload className="w-4 h-4" />
              Upload
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                transition-colors duration-200
                ${viewMode === 'list'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <List className="w-4 h-4" />
              List
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
                transition-colors duration-200
                ${viewMode === 'grid'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <Grid className="w-4 h-4" />
              Grid
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto bg-gray-50">
        {viewMode === 'upload' && (
          <div className="max-w-4xl mx-auto p-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Upload Documents
              </h3>
              <DocumentUpload 
                onUploadComplete={handleUploadComplete}
                onError={handleUploadError}
              />

              {uploadedDocuments.length > 0 && (
                <div className="mt-6 p-4 bg-green-50 rounded-lg">
                  <p className="text-sm font-medium text-green-800">
                    Successfully uploaded {uploadedDocuments.length} document(s)
                  </p>
                </div>
              )}
            </div>

            {/* Upload Instructions */}
            <div className="mt-6 bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Upload Guidelines
              </h3>
              <div className="space-y-3 text-sm text-gray-600">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center flex-shrink-0 text-xs font-semibold">
                    1
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Supported Formats</p>
                    <p>Currently only PDF files are supported. Each file can be up to 100MB.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center flex-shrink-0 text-xs font-semibold">
                    2
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Automatic Processing</p>
                    <p>Documents are automatically split, classified, and indexed for search.</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center flex-shrink-0 text-xs font-semibold">
                    3
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">Privacy Protected</p>
                    <p>All processing happens locally based on your privacy settings.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {viewMode === 'list' && (
          <div className="p-6">
            <div className="bg-white rounded-lg shadow-sm p-8 text-center">
              <FolderOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Document List View
              </h3>
              <p className="text-gray-600">
                Document list view will be implemented here.
              </p>
            </div>
          </div>
        )}

        {viewMode === 'grid' && (
          <div className="p-6">
            <div className="bg-white rounded-lg shadow-sm p-8 text-center">
              <Grid className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Document Grid View
              </h3>
              <p className="text-gray-600">
                Document grid view will be implemented here.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}