import React, { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import { 
  FileText, 
  Calendar, 
  Users, 
  Hash, 
  ChevronUp, 
  ChevronDown, 
  Loader2,
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
  Eye,
  Trash2,
  MoreVertical,
  X,
  Check
} from 'lucide-react'

// Document type enum matching backend
export enum DocumentType {
  EMAIL = "email",
  RFI = "rfi",
  RFP = "rfp",
  CHANGE_ORDER = "change_order",
  SUBMITTAL = "submittal",
  INVOICE = "invoice",
  CONTRACT = "contract",
  DRAWING = "drawing",
  SPECIFICATION = "specification",
  MEETING_MINUTES = "meeting_minutes",
  DAILY_REPORT = "daily_report",
  LETTER = "letter",
  MEMORANDUM = "memorandum",
  PAYMENT_APPLICATION = "payment_application",
  SCHEDULE = "schedule",
  UNKNOWN = "unknown"
}

// Interfaces
interface DocumentMetadata {
  dates: string[]
  parties: string[]
  amounts: number[]
  reference_numbers?: Record<string, string>
}

interface Document {
  id: string
  title: string
  type: DocumentType
  page_count: number
  created_at: string | null
  metadata: DocumentMetadata | null
}

interface DocumentListResponse {
  documents: Document[]
  total: number
  limit: number
  offset: number
}

interface FilterState {
  documentTypes: DocumentType[]
  dateRange: {
    start: Date | null
    end: Date | null
  }
  searchText: string
  selectedParty: string | null
}

type SortField = 'created_at' | 'type' | 'title' | 'page_count'
type SortOrder = 'asc' | 'desc'

interface SortState {
  field: SortField
  order: SortOrder
}

// Document type badge configuration
const documentTypeBadgeConfig: Record<DocumentType, { bg: string; text: string; label: string }> = {
  [DocumentType.EMAIL]: { bg: 'bg-gray-100', text: 'text-gray-700', label: 'Email' },
  [DocumentType.RFI]: { bg: 'bg-blue-100', text: 'text-blue-700', label: 'RFI' },
  [DocumentType.RFP]: { bg: 'bg-purple-100', text: 'text-purple-700', label: 'RFP' },
  [DocumentType.CHANGE_ORDER]: { bg: 'bg-orange-100', text: 'text-orange-700', label: 'Change Order' },
  [DocumentType.SUBMITTAL]: { bg: 'bg-teal-100', text: 'text-teal-700', label: 'Submittal' },
  [DocumentType.INVOICE]: { bg: 'bg-green-100', text: 'text-green-700', label: 'Invoice' },
  [DocumentType.CONTRACT]: { bg: 'bg-indigo-100', text: 'text-indigo-700', label: 'Contract' },
  [DocumentType.DRAWING]: { bg: 'bg-cyan-100', text: 'text-cyan-700', label: 'Drawing' },
  [DocumentType.SPECIFICATION]: { bg: 'bg-pink-100', text: 'text-pink-700', label: 'Specification' },
  [DocumentType.MEETING_MINUTES]: { bg: 'bg-yellow-100', text: 'text-yellow-700', label: 'Meeting Minutes' },
  [DocumentType.DAILY_REPORT]: { bg: 'bg-amber-100', text: 'text-amber-700', label: 'Daily Report' },
  [DocumentType.LETTER]: { bg: 'bg-slate-100', text: 'text-slate-700', label: 'Letter' },
  [DocumentType.MEMORANDUM]: { bg: 'bg-stone-100', text: 'text-stone-700', label: 'Memorandum' },
  [DocumentType.PAYMENT_APPLICATION]: { bg: 'bg-emerald-100', text: 'text-emerald-700', label: 'Payment Application' },
  [DocumentType.SCHEDULE]: { bg: 'bg-rose-100', text: 'text-rose-700', label: 'Schedule' },
  [DocumentType.UNKNOWN]: { bg: 'bg-gray-100', text: 'text-gray-500', label: 'Unknown' }
}

export const DocumentList: React.FC = () => {
  // State
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [totalDocuments, setTotalDocuments] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [showFilters, setShowFilters] = useState(false)
  const [deletingDocuments, setDeletingDocuments] = useState<Set<string>>(new Set())
  const [dropdownOpen, setDropdownOpen] = useState<string | null>(null)
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set())
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const [showBulkDeleteDialog, setShowBulkDeleteDialog] = useState(false)
  
  const documentsPerPage = 20
  
  // Sort state
  const [sortState, setSortState] = useState<SortState>({
    field: 'created_at',
    order: 'desc'
  })
  
  // Filter state
  const [filters, setFilters] = useState<FilterState>({
    documentTypes: [],
    dateRange: {
      start: null,
      end: null
    },
    searchText: '',
    selectedParty: null
  })

  // Extract all unique parties from documents
  const allParties = useMemo(() => {
    const parties = new Set<string>()
    documents.forEach(doc => {
      if (doc.metadata?.parties) {
        doc.metadata.parties.forEach(party => parties.add(party))
      }
    })
    return Array.from(parties).sort()
  }, [documents])

  // Fetch documents
  const fetchDocuments = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const offset = (currentPage - 1) * documentsPerPage
      const params: any = {
        limit: documentsPerPage,
        offset: offset
      }
      
      // Add type filter if selected
      if (filters.documentTypes.length === 1) {
        params.document_type = filters.documentTypes[0]
      }
      
      const response = await axios.get<DocumentListResponse>('/api/v1/documents/list', { params })
      
      let filteredDocs = response.data.documents
      
      // Client-side filtering (since backend doesn't support all filters yet)
      // Filter by multiple document types
      if (filters.documentTypes.length > 1) {
        filteredDocs = filteredDocs.filter(doc => 
          filters.documentTypes.includes(doc.type)
        )
      }
      
      // Filter by date range
      if (filters.dateRange.start || filters.dateRange.end) {
        filteredDocs = filteredDocs.filter(doc => {
          if (!doc.created_at) return false
          const docDate = new Date(doc.created_at)
          if (filters.dateRange.start && docDate < filters.dateRange.start) return false
          if (filters.dateRange.end && docDate > filters.dateRange.end) return false
          return true
        })
      }
      
      // Filter by search text
      if (filters.searchText) {
        const searchLower = filters.searchText.toLowerCase()
        filteredDocs = filteredDocs.filter(doc => 
          doc.title.toLowerCase().includes(searchLower)
        )
      }
      
      // Filter by party
      if (filters.selectedParty) {
        filteredDocs = filteredDocs.filter(doc => 
          doc.metadata?.parties?.includes(filters.selectedParty!)
        )
      }
      
      // Client-side sorting
      filteredDocs.sort((a, b) => {
        let aVal: any, bVal: any
        
        switch (sortState.field) {
          case 'created_at':
            aVal = a.created_at ? new Date(a.created_at).getTime() : 0
            bVal = b.created_at ? new Date(b.created_at).getTime() : 0
            break
          case 'type':
            aVal = a.type
            bVal = b.type
            break
          case 'title':
            aVal = a.title.toLowerCase()
            bVal = b.title.toLowerCase()
            break
          case 'page_count':
            aVal = a.page_count
            bVal = b.page_count
            break
        }
        
        if (sortState.order === 'asc') {
          return aVal > bVal ? 1 : -1
        } else {
          return aVal < bVal ? 1 : -1
        }
      })
      
      setDocuments(filteredDocs)
      setTotalDocuments(response.data.total)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load documents')
      console.error('Error fetching documents:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [currentPage, filters.documentTypes.length === 1 ? filters.documentTypes[0] : null])

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => setDropdownOpen(null)
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [])

  // Sort handler
  const handleSort = (field: SortField) => {
    setSortState(prev => ({
      field,
      order: prev.field === field && prev.order === 'desc' ? 'asc' : 'desc'
    }))
  }

  // Filter handlers
  const toggleDocumentType = (type: DocumentType) => {
    setFilters(prev => ({
      ...prev,
      documentTypes: prev.documentTypes.includes(type)
        ? prev.documentTypes.filter(t => t !== type)
        : [...prev.documentTypes, type]
    }))
    setCurrentPage(1)
  }

  const clearFilters = () => {
    setFilters({
      documentTypes: [],
      dateRange: { start: null, end: null },
      searchText: '',
      selectedParty: null
    })
    setCurrentPage(1)
  }

  // Bulk selection handlers
  const toggleDocumentSelection = (documentId: string) => {
    setSelectedDocuments(prev => {
      const newSet = new Set(prev)
      if (newSet.has(documentId)) {
        newSet.delete(documentId)
      } else {
        newSet.add(documentId)
      }
      return newSet
    })
  }

  const selectAllDocuments = () => {
    setSelectedDocuments(new Set(documents.map(doc => doc.id)))
  }

  const clearSelection = () => {
    setSelectedDocuments(new Set())
  }

  const isAllSelected = documents.length > 0 && selectedDocuments.size === documents.length
  const isIndeterminate = selectedDocuments.size > 0 && selectedDocuments.size < documents.length

  // Document actions
  const viewDocument = (documentId: string) => {
    window.open(`/api/v1/storage/documents/${documentId}/pdf`, '_blank')
  }

  const deleteDocument = async (documentId: string, documentTitle: string) => {
    if (!confirm(`Are you sure you want to delete "${documentTitle}"?\n\nThis action cannot be undone.`)) {
      return
    }

    setDeletingDocuments(prev => new Set(prev).add(documentId))
    setDropdownOpen(null)

    try {
      await axios.delete(`/api/v1/storage/documents/${documentId}`)
      
      // Remove document from local state
      setDocuments(prev => prev.filter(doc => doc.id !== documentId))
      setTotalDocuments(prev => prev - 1)
      
      // If this was the last document on the page and we're not on page 1, go back a page
      if (documents.length === 1 && currentPage > 1) {
        setCurrentPage(prev => prev - 1)
      } else {
        // Refresh to get updated list
        fetchDocuments()
      }
    } catch (err: any) {
      alert(`Failed to delete document: ${err.response?.data?.detail || err.message}`)
      console.error('Error deleting document:', err)
    } finally {
      setDeletingDocuments(prev => {
        const newSet = new Set(prev)
        newSet.delete(documentId)
        return newSet
      })
    }
  }

  const bulkDeleteDocuments = async () => {
    const selectedIds = Array.from(selectedDocuments)
    if (selectedIds.length === 0) return

    setBulkDeleting(true)
    setShowBulkDeleteDialog(false)

    try {
      const response = await axios.delete('/api/v1/storage/documents/bulk', {
        data: {
          document_ids: selectedIds,
          delete_files: true
        }
      })

      const result = response.data
      
      // Remove successfully deleted documents from local state
      if (result.successful_deletions > 0) {
        setDocuments(prev => prev.filter(doc => !selectedIds.includes(doc.id)))
        setTotalDocuments(prev => prev - result.successful_deletions)
        setSelectedDocuments(new Set())
      }

      // Show results
      if (result.failed_deletions > 0) {
        const errorMessages = result.errors.map((err: any) => `${err.document_id}: ${err.error}`).join('\n')
        alert(`Bulk delete completed with some errors:\n\nSuccessful: ${result.successful_deletions}\nFailed: ${result.failed_deletions}\n\nErrors:\n${errorMessages}`)
      } else {
        alert(`Successfully deleted ${result.successful_deletions} documents`)
      }

      // Refresh the list
      fetchDocuments()
    } catch (err: any) {
      alert(`Failed to delete documents: ${err.response?.data?.detail || err.message}`)
      console.error('Error bulk deleting documents:', err)
    } finally {
      setBulkDeleting(false)
    }
  }

  // Pagination
  const totalPages = Math.ceil(totalDocuments / documentsPerPage)

  // Format date for display
  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '-'
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    })
  }

  // Extract reference numbers
  const getReferenceNumber = (doc: Document): string => {
    if (!doc.metadata?.reference_numbers) return '-'
    const refs = Object.entries(doc.metadata.reference_numbers)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ')
    return refs || '-'
  }

  // Sort icon component
  const SortIcon: React.FC<{ field: SortField; currentField: SortField; order: SortOrder }> = ({ 
    field, 
    currentField, 
    order 
  }) => {
    if (field !== currentField) {
      return <div className="w-4 h-4" />
    }
    return order === 'desc' ? (
      <ChevronDown className="w-4 h-4" />
    ) : (
      <ChevronUp className="w-4 h-4" />
    )
  }

  // Loading state
  if (loading && documents.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading documents...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error && documents.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-gray-900 font-medium mb-2">Error loading documents</p>
          <p className="text-gray-600 text-sm">{error}</p>
          <button
            onClick={fetchDocuments}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6">
      {/* Header with filters toggle */}
      <div className="mb-6 flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Documents ({totalDocuments})
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Construction litigation documents from uploaded PDFs
          </p>
        </div>
        
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg border
            transition-colors duration-200
            ${showFilters 
              ? 'bg-blue-50 border-blue-200 text-blue-700' 
              : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
            }
          `}
        >
          <Filter className="w-4 h-4" />
          Filters
          {(filters.documentTypes.length > 0 || filters.searchText || filters.selectedParty) && (
            <span className="ml-1 px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full">
              {filters.documentTypes.length + (filters.searchText ? 1 : 0) + (filters.selectedParty ? 1 : 0)}
            </span>
          )}
        </button>
      </div>

      {/* Bulk actions toolbar */}
      {selectedDocuments.size > 0 && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-blue-900">
                {selectedDocuments.size} document{selectedDocuments.size === 1 ? '' : 's'} selected
              </span>
              <button
                onClick={clearSelection}
                className="text-sm text-blue-700 hover:text-blue-900 flex items-center gap-1"
              >
                <X className="w-4 h-4" />
                Clear selection
              </button>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowBulkDeleteDialog(true)}
                disabled={bulkDeleting}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${bulkDeleting
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-red-600 text-white hover:bg-red-700'
                  }
                `}
              >
                {bulkDeleting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
                Delete {selectedDocuments.size} document{selectedDocuments.size === 1 ? '' : 's'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters panel */}
      {showFilters && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Search */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Search
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={filters.searchText}
                  onChange={(e) => {
                    setFilters(prev => ({ ...prev, searchText: e.target.value }))
                    setCurrentPage(1)
                  }}
                  placeholder="Search titles..."
                  className="pl-10 pr-3 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            {/* Document types */}
            <div className="lg:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Document Types
              </label>
              <div className="flex flex-wrap gap-2">
                {Object.entries(documentTypeBadgeConfig).map(([type, config]) => (
                  <button
                    key={type}
                    onClick={() => toggleDocumentType(type as DocumentType)}
                    className={`
                      px-3 py-1 rounded-full text-xs font-medium transition-all
                      ${filters.documentTypes.includes(type as DocumentType)
                        ? `${config.bg} ${config.text} ring-2 ring-offset-1 ring-${config.text.split('-')[1]}-500`
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }
                    `}
                  >
                    {config.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Party filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Party
              </label>
              <select
                value={filters.selectedParty || ''}
                onChange={(e) => {
                  setFilters(prev => ({ ...prev, selectedParty: e.target.value || null }))
                  setCurrentPage(1)
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">All parties</option>
                {allParties.map(party => (
                  <option key={party} value={party}>{party}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Clear filters */}
          {(filters.documentTypes.length > 0 || filters.searchText || filters.selectedParty) && (
            <div className="mt-4 flex justify-end">
              <button
                onClick={clearFilters}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Clear all filters
              </button>
            </div>
          )}
        </div>
      )}

      {/* Documents table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-6 py-3 text-left w-12">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={isAllSelected}
                      ref={(input) => {
                        if (input) input.indeterminate = isIndeterminate
                      }}
                      onChange={() => {
                        if (isAllSelected || isIndeterminate) {
                          clearSelection()
                        } else {
                          selectAllDocuments()
                        }
                      }}
                      className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    />
                  </div>
                </th>
                <th className="px-6 py-3 text-left">
                  <button
                    onClick={() => handleSort('title')}
                    className="flex items-center gap-1 text-xs font-medium text-gray-700 uppercase tracking-wider hover:text-gray-900"
                  >
                    Title
                    <SortIcon field="title" currentField={sortState.field} order={sortState.order} />
                  </button>
                </th>
                <th className="px-6 py-3 text-left">
                  <button
                    onClick={() => handleSort('type')}
                    className="flex items-center gap-1 text-xs font-medium text-gray-700 uppercase tracking-wider hover:text-gray-900"
                  >
                    Type
                    <SortIcon field="type" currentField={sortState.field} order={sortState.order} />
                  </button>
                </th>
                <th className="px-6 py-3 text-left">
                  <button
                    onClick={() => handleSort('created_at')}
                    className="flex items-center gap-1 text-xs font-medium text-gray-700 uppercase tracking-wider hover:text-gray-900"
                  >
                    Date
                    <SortIcon field="created_at" currentField={sortState.field} order={sortState.order} />
                  </button>
                </th>
                <th className="px-6 py-3 text-left">
                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Parties
                  </span>
                </th>
                <th className="px-6 py-3 text-left">
                  <button
                    onClick={() => handleSort('page_count')}
                    className="flex items-center gap-1 text-xs font-medium text-gray-700 uppercase tracking-wider hover:text-gray-900"
                  >
                    Pages
                    <SortIcon field="page_count" currentField={sortState.field} order={sortState.order} />
                  </button>
                </th>
                <th className="px-6 py-3 text-left">
                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Reference
                  </span>
                </th>
                <th className="px-6 py-3 text-center">
                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wider">
                    Actions
                  </span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {documents.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-6 py-12 text-center">
                    <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-900 font-medium mb-2">No documents found</p>
                    <p className="text-gray-600 text-sm">
                      {filters.documentTypes.length > 0 || filters.searchText || filters.selectedParty
                        ? 'Try adjusting your filters'
                        : 'Upload some PDFs to get started'
                      }
                    </p>
                  </td>
                </tr>
              ) : (
                documents.map((doc) => {
                  const badgeConfig = documentTypeBadgeConfig[doc.type]
                  const isDeleting = deletingDocuments.has(doc.id)
                  
                  return (
                    <tr
                      key={doc.id}
                      className={`hover:bg-gray-50 transition-colors ${selectedDocuments.has(doc.id) ? 'bg-blue-50' : ''}`}
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedDocuments.has(doc.id)}
                            onChange={() => toggleDocumentSelection(doc.id)}
                            className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                          />
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <FileText className="w-5 h-5 text-gray-400 flex-shrink-0" />
                          <span className="text-sm font-medium text-gray-900 truncate max-w-xs">
                            {doc.title}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`
                          inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                          ${badgeConfig.bg} ${badgeConfig.text}
                        `}>
                          {badgeConfig.label}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">
                        {formatDate(doc.created_at)}
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm text-gray-600">
                          {doc.metadata?.parties?.length ? (
                            <div className="flex items-center gap-1">
                              <Users className="w-4 h-4 text-gray-400" />
                              <span className="truncate max-w-xs">
                                {doc.metadata.parties.slice(0, 2).join(', ')}
                                {doc.metadata.parties.length > 2 && (
                                  <span className="text-gray-400">
                                    {' '}+{doc.metadata.parties.length - 2}
                                  </span>
                                )}
                              </span>
                            </div>
                          ) : (
                            <span className="text-gray-400">-</span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-600">
                        {doc.page_count}
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm text-gray-600">
                          {getReferenceNumber(doc) !== '-' ? (
                            <div className="flex items-center gap-1">
                              <Hash className="w-4 h-4 text-gray-400" />
                              <span className="truncate max-w-xs">
                                {getReferenceNumber(doc)}
                              </span>
                            </div>
                          ) : (
                            <span className="text-gray-400">-</span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center justify-center gap-2">
                          <button
                            onClick={() => viewDocument(doc.id)}
                            className="p-1.5 rounded-lg text-gray-500 hover:text-blue-600 hover:bg-blue-50 transition-colors"
                            title="View PDF"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          
                          <div className="relative">
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setDropdownOpen(dropdownOpen === doc.id ? null : doc.id)
                              }}
                              disabled={isDeleting}
                              className={`
                                p-1.5 rounded-lg transition-colors
                                ${isDeleting
                                  ? 'text-gray-300 cursor-not-allowed'
                                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                                }
                              `}
                              title="More actions"
                            >
                              {isDeleting ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                              ) : (
                                <MoreVertical className="w-4 h-4" />
                              )}
                            </button>
                            
                            {dropdownOpen === doc.id && !isDeleting && (
                              <div 
                                className="absolute right-0 top-8 mt-1 w-32 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-10"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <button
                                  onClick={() => deleteDocument(doc.id, doc.title)}
                                  className="flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 w-full text-left"
                                >
                                  <Trash2 className="w-4 h-4" />
                                  Delete
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Showing {((currentPage - 1) * documentsPerPage) + 1} to{' '}
              {Math.min(currentPage * documentsPerPage, totalDocuments)} of {totalDocuments} documents
            </div>
            
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
                className={`
                  p-2 rounded-lg transition-colors
                  ${currentPage === 1
                    ? 'text-gray-400 cursor-not-allowed'
                    : 'text-gray-700 hover:bg-gray-100'
                  }
                `}
              >
                <ChevronLeft className="w-5 h-5" />
              </button>
              
              <div className="flex items-center gap-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum
                  if (totalPages <= 5) {
                    pageNum = i + 1
                  } else if (currentPage <= 3) {
                    pageNum = i + 1
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i
                  } else {
                    pageNum = currentPage - 2 + i
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`
                        px-3 py-1 rounded-lg text-sm font-medium transition-colors
                        ${pageNum === currentPage
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-700 hover:bg-gray-100'
                        }
                      `}
                    >
                      {pageNum}
                    </button>
                  )
                })}
              </div>
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
                className={`
                  p-2 rounded-lg transition-colors
                  ${currentPage === totalPages
                    ? 'text-gray-400 cursor-not-allowed'
                    : 'text-gray-700 hover:bg-gray-100'
                  }
                `}
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Bulk delete confirmation dialog */}
      {showBulkDeleteDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex-shrink-0 w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 className="w-4 h-4 text-red-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">
                Delete Selected Documents
              </h3>
            </div>
            
            <div className="mb-6">
              <p className="text-sm text-gray-600 mb-4">
                Are you sure you want to delete the following {selectedDocuments.size} document{selectedDocuments.size === 1 ? '' : 's'}? 
                This action cannot be undone.
              </p>
              
              <div className="max-h-32 overflow-y-auto bg-gray-50 rounded-lg p-3">
                <ul className="space-y-1">
                  {documents
                    .filter(doc => selectedDocuments.has(doc.id))
                    .map(doc => (
                      <li key={doc.id} className="text-sm text-gray-700 flex items-center gap-2">
                        <FileText className="w-3 h-3 text-gray-400 flex-shrink-0" />
                        <span className="truncate">{doc.title}</span>
                      </li>
                    ))
                  }
                </ul>
              </div>
            </div>
            
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowBulkDeleteDialog(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={bulkDeleteDocuments}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete {selectedDocuments.size} Document{selectedDocuments.size === 1 ? '' : 's'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}