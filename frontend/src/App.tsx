import { useState, useEffect } from 'react'
import { Shield, ShieldAlert, Globe } from 'lucide-react'
import axios from 'axios'
import { DocumentBrowser } from './components/document-browser'

type PrivacyMode = 'full_local' | 'hybrid_safe' | 'full_featured'

interface PrivacyInfo {
  mode: PrivacyMode
  description: string
  local_only: boolean
  api_enabled: boolean
}

function App() {
  const [privacyInfo, setPrivacyInfo] = useState<PrivacyInfo | null>(null)
  const [health, setHealth] = useState<any>(null)
  const [activeView, setActiveView] = useState<'home' | 'documents'>('home')

  useEffect(() => {
    // Fetch initial status
    fetchHealth()
    fetchPrivacyStatus()
  }, [])

  const fetchHealth = async () => {
    try {
      const response = await axios.get('/api/v1')
      setHealth(response.data)
    } catch (error) {
      console.error('Failed to fetch health:', error)
    }
  }

  const fetchPrivacyStatus = async () => {
    try {
      const response = await axios.get('/api/v1/privacy')
      setPrivacyInfo(response.data)
    } catch (error) {
      console.error('Failed to fetch privacy status:', error)
    }
  }

  const getPrivacyIcon = () => {
    if (!privacyInfo) return null
    
    switch (privacyInfo.mode) {
      case 'full_local':
        return <Shield className="w-5 h-5 text-green-600" />
      case 'hybrid_safe':
        return <ShieldAlert className="w-5 h-5 text-yellow-600" />
      case 'full_featured':
        return <Globe className="w-5 h-5 text-blue-600" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900 cursor-pointer" onClick={() => setActiveView('home')}>CLaiM</h1>
              <span className="ml-3 text-sm text-gray-500">Construction Litigation AI Manager</span>
            </div>
            
            <div className="flex items-center gap-4">
              <nav className="flex gap-4 mr-6">
                <button
                  onClick={() => setActiveView('home')}
                  className={`text-sm font-medium transition-colors ${
                    activeView === 'home' 
                      ? 'text-blue-600' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Home
                </button>
                <button
                  onClick={() => setActiveView('documents')}
                  className={`text-sm font-medium transition-colors ${
                    activeView === 'documents' 
                      ? 'text-blue-600' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Documents
                </button>
              </nav>
              
              {privacyInfo && (
                <div className="flex items-center gap-2 px-3 py-1 bg-gray-100 rounded-lg">
                  {getPrivacyIcon()}
                  <span className="text-sm font-medium text-gray-700">
                    {privacyInfo.mode.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        {activeView === 'home' ? (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Welcome to CLaiM</h2>
              
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-medium text-blue-900 mb-2">Getting Started</h3>
                  <p className="text-blue-700">
                    CLaiM helps you analyze construction litigation documents with AI-powered insights
                    while maintaining complete privacy control.
                  </p>
                  <button
                    onClick={() => setActiveView('documents')}
                    className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Upload Documents
                  </button>
                </div>

                {privacyInfo && (
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <h3 className="font-medium text-gray-900 mb-2">Privacy Mode</h3>
                    <p className="text-gray-700">{privacyInfo.description}</p>
                  </div>
                )}

                {health && (
                  <div className="p-4 bg-green-50 rounded-lg">
                    <h3 className="font-medium text-green-900 mb-2">System Status</h3>
                    <p className="text-green-700">
                      API Version: {health.version} • Documentation: <a href={health.docs} className="underline">API Docs</a>
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <DocumentBrowser />
        )}
      </main>
    </div>
  )
}

export default App