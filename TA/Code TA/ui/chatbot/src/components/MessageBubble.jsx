import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ConfidenceBadge from './ConfidenceBadge'

/**
 * Avatar untuk user dan assistant.
 */
function Avatar({ role }) {
  if (role === 'user') {
    return (
      <div
        className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center
                   text-slate-600 font-medium text-sm"
        aria-label="You"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
            clipRule="evenodd"
          />
        </svg>
      </div>
    )
  }
  return (
    <div
      className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-slate-800 to-slate-900
                 flex items-center justify-center text-white text-sm"
      aria-label="BioRAG"
    >
      <span aria-hidden="true">🧬</span>
    </div>
  )
}

/**
 * Bubble pesan tunggal.
 *
 * Props:
 *   role     — 'user' atau 'assistant'
 *   content  — teks pesan (markdown didukung untuk assistant)
 *   source     — { pubid, topic } opsional, ditampilkan pada pesan assistant
 *   confidence — { score, level, refused, breakdown, thresholds } from backend
 *   children   — optional override content (mis. untuk TypingIndicator)
 */
export default function MessageBubble({ role, content, source, confidence, children }) {
  const isUser = role === 'user'

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} fade-in`}>
      <Avatar role={role} />

      <div className={`flex flex-col max-w-[85%] sm:max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Label role + confidence badge (assistant only) */}
        <div className="flex items-center gap-2 mb-1 px-1 flex-wrap">
          <span className="text-[11px] font-medium text-slate-500">
            {isUser ? 'You' : 'BioRAG'}
          </span>
          {!isUser && confidence && <ConfidenceBadge confidence={confidence} />}
        </div>

        {/* Konten pesan */}
        <div
          className={`px-4 py-3 rounded-2xl text-[15px] leading-relaxed
                      ${
                        isUser
                          ? 'bg-slate-900 text-white rounded-br-sm'
                          : 'bg-slate-100 text-slate-900 rounded-bl-sm markdown-body'
                      }`}
        >
          {children ? (
            children
          ) : isUser ? (
            <p className="whitespace-pre-wrap">{content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          )}
        </div>

        {/* Source tag (untuk pesan assistant yang punya sumber) */}
        {!isUser && source && (
          <div className="mt-2 px-1 flex items-center gap-1.5 text-[11px] text-slate-500">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-3 h-3"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
                clipRule="evenodd"
              />
            </svg>
            <span>
              Source: PubMed ID{' '}
              <a
                href={`https://pubmed.ncbi.nlm.nih.gov/${source.pubid}/`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                {source.pubid}
              </a>{' '}
              <span className="text-slate-400">·</span> {source.topic}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
