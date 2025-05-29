import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, X, AlertCircle, CheckCircle, Loader2 } from 'lucide-react'
import axios from 'axios'

interface UploadedFile {
  id: string
  file: File
  status: 'pending' | 'uploading' | 'processing' | 'success' | 'error'
  progress: number
  error?: string
  documentId?: string
}

interface DocumentUploadProps {
  onUploadComplete?: (documentId: string) => void
  onError?: (error: string) => void
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ 
  onUploadComplete, 
  onError 
}) => {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isUploading, setIsUploading] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'pending',
      progress: 0
    }))
    setFiles(prev => [...prev, ...newFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true
  })

  const uploadFile = async (uploadedFile: UploadedFile) => {
    const formData = new FormData()
    formData.append('file', uploadedFile.file)

    try {
      // Update status to uploading
      setFiles(prev => prev.map(f => 
        f.id === uploadedFile.id ? { ...f, status: 'uploading' } : f
      ))

      // Upload file
      const response = await axios.post('/api/v1/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total 
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0
          
          setFiles(prev => prev.map(f => 
            f.id === uploadedFile.id ? { ...f, progress } : f
          ))
        }
      })

      // Update status to processing
      setFiles(prev => prev.map(f => 
        f.id === uploadedFile.id ? { ...f, status: 'processing', progress: 100 } : f
      ))

      // Simulate processing time (in real app, you'd poll for status)
      setTimeout(() => {
        setFiles(prev => prev.map(f => 
          f.id === uploadedFile.id 
            ? { ...f, status: 'success', documentId: response.data.document_id } 
            : f
        ))
        
        if (onUploadComplete && response.data.document_id) {
          onUploadComplete(response.data.document_id)
        }
      }, 2000)

    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Upload failed'
      setFiles(prev => prev.map(f => 
        f.id === uploadedFile.id 
          ? { ...f, status: 'error', error: errorMessage } 
          : f
      ))
      
      if (onError) {
        onError(errorMessage)
      }
    }
  }

  const handleUpload = async () => {
    const pendingFiles = files.filter(f => f.status === 'pending')
    if (pendingFiles.length === 0) return

    setIsUploading(true)

    // Upload files sequentially (could be parallel in production)
    for (const file of pendingFiles) {
      await uploadFile(file)
    }

    setIsUploading(false)
  }

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id))
  }

  const clearCompleted = () => {
    setFiles(prev => prev.filter(f => f.status !== 'success'))
  }

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'pending':
        return <FileText className="w-5 h-5 text-gray-400" />
      case 'uploading':
      case 'processing':
        return <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-600" />
    }
  }

  const getStatusText = (file: UploadedFile) => {
    switch (file.status) {
      case 'pending':
        return 'Ready to upload'
      case 'uploading':
        return `Uploading... ${file.progress}%`
      case 'processing':
        return 'Processing document...'
      case 'success':
        return 'Upload complete'
      case 'error':
        return file.error || 'Upload failed'
    }
  }

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400 bg-gray-50'
          }
        `}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
        
        {isDragActive ? (
          <p className="text-lg text-blue-600 font-medium">Drop PDFs here...</p>
        ) : (
          <div>
            <p className="text-lg text-gray-700 font-medium mb-2">
              Drag & drop PDF files here
            </p>
            <p className="text-sm text-gray-500">
              or click to browse for files
            </p>
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div className="mt-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              Files ({files.length})
            </h3>
            {files.some(f => f.status === 'success') && (
              <button
                onClick={clearCompleted}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Clear completed
              </button>
            )}
          </div>

          <div className="space-y-2">
            {files.map(file => (
              <div
                key={file.id}
                className="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200"
              >
                <div className="flex items-center gap-3 flex-1">
                  {getStatusIcon(file.status)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(file.file.size / 1024 / 1024).toFixed(2)} MB • {getStatusText(file)}
                    </p>
                  </div>
                </div>

                {file.status === 'pending' && (
                  <button
                    onClick={() => removeFile(file.id)}
                    className="ml-3 text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-5 h-5" />
                  </button>
                )}

                {file.status === 'uploading' && (
                  <div className="ml-3 w-32">
                    <div className="bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${file.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {files.some(f => f.status === 'pending') && (
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className={`
                mt-4 w-full py-2 px-4 rounded-lg font-medium
                transition-colors duration-200
                ${isUploading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
                }
              `}
            >
              {isUploading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </span>
              ) : (
                `Upload ${files.filter(f => f.status === 'pending').length} file(s)`
              )}
            </button>
          )}
        </div>
      )}
    </div>
  )
}