import React, { useEffect } from 'react'

export default function DocumentModal({ doc, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  if (!doc) return null

  // Highlight the relevant snippet inside the full content
  const renderHighlightedContent = () => {
    if (!doc.relevantSnippet || !doc.content.includes(doc.relevantSnippet)) {
      return <p className="text-gray-300 text-sm leading-relaxed">{doc.content}</p>
    }

    const parts = doc.content.split(doc.relevantSnippet)
    return (
      <p className="text-gray-300 text-sm leading-relaxed">
        {parts[0]}
        <mark className="bg-yellow-400/30 text-yellow-200 px-0.5 rounded">
          {doc.relevantSnippet}
        </mark>
        {parts[1]}
      </p>
    )
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: 'rgba(0,0,0,0.65)' }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-gray-700 gap-4">
          <div className="flex-1 min-w-0">
            <h2 className="text-white font-semibold text-base leading-snug">
              {doc.title}
            </h2>
            <p className="text-gray-400 text-xs mt-1">{doc.journal}</p>
          </div>
          <button
            onClick={onClose}
            className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
            aria-label="Close modal"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-4 h-4"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        {/* Scores row */}
        <div className="flex items-center gap-4 px-6 py-3 border-b border-gray-700/50 bg-gray-900/50">
          {doc.bm25Score !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">BM25</span>
              <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${doc.bm25Score * 100}%` }}
                />
              </div>
              <span className="text-blue-400 text-xs font-mono">
                {doc.bm25Score.toFixed(3)}
              </span>
            </div>
          )}
          {doc.ceScore !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">CE Score</span>
              <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500 rounded-full"
                  style={{ width: `${doc.ceScore * 100}%` }}
                />
              </div>
              <span className="text-violet-400 text-xs font-mono">
                {doc.ceScore.toFixed(3)}
              </span>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Relevant Snippet callout */}
          {doc.relevantSnippet && (
            <div className="bg-yellow-400/10 border border-yellow-400/30 rounded-lg p-3">
              <p className="text-yellow-300 text-xs font-semibold mb-1 uppercase tracking-wide">
                Key Snippet
              </p>
              <p className="text-yellow-100 text-sm italic">
                &ldquo;{doc.relevantSnippet}&rdquo;
              </p>
            </div>
          )}

          {/* Full text */}
          <div>
            <p className="text-gray-500 text-xs font-semibold uppercase tracking-wide mb-2">
              Full Abstract
            </p>
            {renderHighlightedContent()}
          </div>
        </div>
      </div>
    </div>
  )
}
