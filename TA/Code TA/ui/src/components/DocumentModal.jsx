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
              Dokumen {doc.id}
            </h2>
            <p className="text-gray-400 text-xs mt-1">
              Konteks dari PubMedQA
            </p>
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
        <div className="flex flex-wrap items-center gap-4 px-6 py-3 border-b border-gray-700/50 bg-gray-900/50">
          {doc.bm25_score !== null && doc.bm25_score !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">BM25</span>
              <span className="text-blue-400 text-xs font-mono font-bold">
                {doc.bm25_score.toFixed(2)}
              </span>
            </div>
          )}
          {doc.dense_score !== null && doc.dense_score !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">Dense (cos-sim)</span>
              <span className="text-emerald-400 text-xs font-mono font-bold">
                {doc.dense_score.toFixed(3)}
              </span>
            </div>
          )}
          {doc.rrf_score !== null && doc.rrf_score !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">RRF</span>
              <span className="text-amber-400 text-xs font-mono font-bold">
                {doc.rrf_score.toFixed(4)}
              </span>
            </div>
          )}
          {doc.reranker_score !== null && doc.reranker_score !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">CrossEncoder</span>
              <span className="text-teal-400 text-xs font-mono font-bold">
                {doc.reranker_score.toFixed(3)}
              </span>
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          <div>
            <p className="text-gray-500 text-xs font-semibold uppercase tracking-wide mb-2">
              Isi Dokumen
            </p>
            <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">
              {doc.content}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
