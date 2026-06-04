import React from 'react'

/**
 * Top bar of the main content column.
 *
 * Holds the BioRAG logo + (on small screens) toggle buttons for the left
 * sidebar and the right pipeline panel.
 *
 * Props:
 *   onToggleSidebar    — open/close left drawer (mobile only)
 *   onTogglePipeline   — open/close right drawer (mobile only)
 *   showPipelineToggle — hide the right-toggle when there's no pipeline data
 *   activeTitle        — current conversation title to display center-top
 */
export default function Header({
  onToggleSidebar,
  onTogglePipeline,
  showPipelineToggle,
  activeTitle,
}) {
  return (
    <header className="flex-shrink-0 border-b border-slate-200 bg-white">
      <div className="px-4 sm:px-6 py-3 flex items-center justify-between gap-3">
        {/* Left: sidebar toggle (mobile) + logo */}
        <div className="flex items-center gap-2 min-w-0">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="lg:hidden p-1.5 rounded text-slate-600 hover:bg-slate-100"
            aria-label="Open conversations"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className="w-8 h-8 rounded-xl bg-gradient-to-br from-slate-800 to-slate-900
                         flex items-center justify-center text-white flex-shrink-0"
            >
              <span aria-hidden="true">🧬</span>
            </div>
            <div className="min-w-0">
              <div className="text-[15px] font-semibold text-slate-900 leading-tight">BioRAG</div>
              <div className="text-[11px] text-slate-500 leading-tight truncate">
                {activeTitle || 'Biomedical Q&A'}
              </div>
            </div>
          </div>
        </div>

        {/* Right: pipeline toggle (mobile) */}
        {showPipelineToggle && (
          <button
            type="button"
            onClick={onTogglePipeline}
            className="xl:hidden p-1.5 rounded text-slate-600 hover:bg-slate-100
                       flex items-center gap-1.5 text-[12px] font-medium"
            aria-label="Show pipeline details"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                clipRule="evenodd"
              />
            </svg>
            <span className="hidden sm:inline">Details</span>
          </button>
        )}
      </div>
    </header>
  )
}
