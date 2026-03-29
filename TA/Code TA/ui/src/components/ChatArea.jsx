import React, { useEffect, useRef } from 'react'
import RAGSteps from './RAGSteps'

const labelConfig = {
  yes: {
    text: 'YES',
    badgeClasses: 'bg-green-400/15 text-green-400 border-green-400/30',
    dotClass: 'bg-green-400',
  },
  maybe: {
    text: 'MAYBE',
    badgeClasses: 'bg-yellow-400/15 text-yellow-400 border-yellow-400/30',
    dotClass: 'bg-yellow-400',
  },
  no: {
    text: 'NO',
    badgeClasses: 'bg-red-400/15 text-red-400 border-red-400/30',
    dotClass: 'bg-red-400',
  },
}

// Spinner for loading states
function Spinner() {
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '0ms' }} />
      <div className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '150ms' }} />
      <div className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  )
}

export default function ChatArea({ activeQuestion, phase, typedAnswer, onDocClick, useQR, useCR, configKey }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom as content appears
  useEffect(() => {
    if (activeQuestion && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [phase, typedAnswer, activeQuestion])

  if (!activeQuestion) {
    // Empty state
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-950">
        <div className="text-center max-w-sm px-6">
          <div className="w-16 h-16 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center mx-auto mb-5">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-8 h-8 text-gray-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
          </div>
          <h3 className="text-white font-semibold text-base mb-2">Select a question to begin</h3>
          <p className="text-gray-500 text-sm leading-relaxed">
            Choose one of the sample PubMedQA questions from the sidebar to visualize the full RAG pipeline in action.
          </p>
          <div className="mt-6 grid grid-cols-3 gap-3">
            {['Query Rewriting', 'BM25 Retrieval', 'Context Reranking'].map((label) => (
              <div
                key={label}
                className="bg-gray-900 border border-gray-800 rounded-lg p-2.5 text-center"
              >
                <p className="text-gray-500 text-xs">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const predictedLabel = activeQuestion.configs[configKey].label
  const lc = labelConfig[predictedLabel]

  const statusText =
    phase === 1 ? (useQR ? 'Rewriting query...' : 'Retrieving documents...') :
    phase === 2 ? 'Retrieving documents...' :
    phase === 3 ? (useCR ? 'Reranking context...' : 'Generating answer...') :
    phase === 4 ? 'Generating answer...' : ''

  return (
    <div className="flex-1 flex flex-col bg-gray-950 overflow-hidden">
      {/* Top bar */}
      <div className="flex-shrink-0 px-6 py-3 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-gray-400 text-xs">Question {activeQuestion.id} of 5</span>
          <span className="text-gray-700">·</span>
          <span className="text-gray-400 text-xs">{activeQuestion.shortLabel}</span>
        </div>
        {phase > 0 && phase < 5 && (
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" />
            <span className="text-violet-400 text-xs">{statusText}</span>
          </div>
        )}
        {phase === 5 && (
          <span className={`text-xs font-semibold border px-2 py-0.5 rounded-full ${lc.badgeClasses}`}>
            {lc.text}
          </span>
        )}
      </div>

      {/* Scrollable chat content */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-3xl mx-auto space-y-2">

          {/* User message bubble */}
          <div className="flex justify-end">
            <div className="max-w-lg bg-gray-800 rounded-2xl rounded-tr-sm px-4 py-3">
              <p className="text-white text-sm leading-relaxed">{activeQuestion.query}</p>
            </div>
          </div>

          {/* RAG Steps inline */}
          {phase >= 1 && (
            <RAGSteps
              question={activeQuestion}
              phase={phase}
              typedAnswer={typedAnswer}
              onDocClick={onDocClick}
              useQR={useQR}
              useCR={useCR}
              configKey={configKey}
            />
          )}

          {/* Assistant answer bubble - shown when done */}
          {phase === 5 && (
            <div className="flex justify-start fade-in">
              <div className="flex items-start gap-3 max-w-2xl">
                {/* Avatar */}
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-600 flex items-center justify-center mt-0.5">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4 text-white"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5 2a1 1 0 011 1v1h1a1 1 0 010 2H6v1a1 1 0 01-2 0V6H3a1 1 0 010-2h1V3a1 1 0 011-1zm0 10a1 1 0 011 1v1h1a1 1 0 110 2H6v1a1 1 0 11-2 0v-1H3a1 1 0 110-2h1v-1a1 1 0 011-1zM12 2a1 1 0 01.967.744L14.146 7.2 17.5 9.134a1 1 0 010 1.732l-3.354 1.935-1.18 4.455a1 1 0 01-1.933 0L9.854 12.8 6.5 10.866a1 1 0 010-1.732l3.354-1.935 1.18-4.455A1 1 0 0112 2z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-white font-medium text-sm">RAG Assistant</span>
                    <span className={`text-xs font-semibold border px-2 py-0.5 rounded-full ${lc.badgeClasses}`}>
                      {lc.text}
                    </span>
                  </div>
                  <div className="bg-gray-900 border border-gray-700/60 rounded-2xl rounded-tl-sm px-4 py-3">
                    <p className="text-gray-200 text-sm leading-relaxed whitespace-pre-line">
                      {activeQuestion.answer}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Thinking indicator while pipeline is running */}
          {phase > 0 && phase < 5 && (
            <div className="flex justify-start">
              <div className="flex items-center gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-600/30 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
                </div>
                <div className="bg-gray-900 border border-gray-700/60 rounded-2xl rounded-tl-sm px-4 py-3">
                  <Spinner />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} className="h-4" />
        </div>
      </div>
    </div>
  )
}
