import React, { useState, useEffect } from 'react'

// ---- Icon components ----

function IconPencil() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
      <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
    </svg>
  )
}

function IconSearch() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
    </svg>
  )
}

function IconSort() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
      <path d="M5 4a1 1 0 00-2 0v7.268a2 2 0 000 3.464V16a1 1 0 102 0v-1.268a2 2 0 000-3.464V4zM11 4a1 1 0 10-2 0v1.268a2 2 0 000 3.464V16a1 1 0 102 0V8.732a2 2 0 000-3.464V4zM16 3a1 1 0 011 1v7.268a2 2 0 010 3.464V16a1 1 0 11-2 0v-1.268a2 2 0 010-3.464V4a1 1 0 011-1z" />
    </svg>
  )
}

function IconSparkles() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M5 2a1 1 0 011 1v1h1a1 1 0 010 2H6v1a1 1 0 01-2 0V6H3a1 1 0 010-2h1V3a1 1 0 011-1zm0 10a1 1 0 011 1v1h1a1 1 0 110 2H6v1a1 1 0 11-2 0v-1H3a1 1 0 110-2h1v-1a1 1 0 011-1zM12 2a1 1 0 01.967.744L14.146 7.2 17.5 9.134a1 1 0 010 1.732l-3.354 1.935-1.18 4.455a1 1 0 01-1.933 0L9.854 12.8 6.5 10.866a1 1 0 010-1.732l3.354-1.935 1.18-4.455A1 1 0 0112 2z" clipRule="evenodd" />
    </svg>
  )
}

// ---- Sub-components ----

function StepCard({ icon, title, badge, children, visible }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    if (visible) {
      const t = setTimeout(() => setMounted(true), 50)
      return () => clearTimeout(t)
    } else {
      setMounted(false)
    }
  }, [visible])

  if (!visible) return null

  return (
    <div
      className={`bg-gray-900 border border-gray-700/60 rounded-xl overflow-hidden transition-all duration-300 ${
        mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
      }`}
    >
      {/* Step header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700/40">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-violet-600/20 text-violet-400 flex items-center justify-center">
            {icon}
          </div>
          <span className="text-white font-medium text-sm">{title}</span>
        </div>
        {badge && (
          <span className="text-xs bg-violet-600/20 text-violet-300 border border-violet-500/30 px-2 py-0.5 rounded-full">
            {badge}
          </span>
        )}
      </div>
      {/* Step content */}
      <div className="p-4">{children}</div>
    </div>
  )
}

// Score bar that animates in
function ScoreBar({ score, color = 'blue' }) {
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => setWidth(score * 100), 100)
    return () => clearTimeout(t)
  }, [score])

  const colorClass = color === 'violet' ? 'bg-violet-500' : 'bg-blue-500'

  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClass} rounded-full score-bar`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`text-xs font-mono flex-shrink-0 ${color === 'violet' ? 'text-violet-400' : 'text-blue-400'}`}>
        {score.toFixed(3)}
      </span>
    </div>
  )
}

// ---- Step 1: Query Rewriting ----
function Step1QueryRewriting({ question, phase }) {
  return (
    <StepCard
      icon={<IconPencil />}
      title="Query Rewriting"
      badge={phase > 1 ? 'Completed' : 'Processing...'}
      visible={phase >= 1}
    >
      <div className="space-y-3">
        <div>
          <p className="text-gray-500 text-xs font-medium mb-1.5 uppercase tracking-wide">Original Query</p>
          <p className="text-gray-500 text-sm line-through leading-relaxed">
            {question.query}
          </p>
        </div>
        {phase >= 2 && (
          <div className="fade-in">
            <div className="flex items-center gap-2 mb-1.5">
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">Rewritten Query</p>
              <span className="text-xs text-violet-400 bg-violet-400/10 px-1.5 py-0.5 rounded">expanded</span>
            </div>
            <p className="text-violet-300 text-sm leading-relaxed border-l-2 border-violet-500/50 pl-3">
              {question.rewrittenQuery}
            </p>
          </div>
        )}
        {phase < 2 && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="w-2 h-2 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="w-2 h-2 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '300ms' }} />
            <span className="text-gray-500 text-xs">Expanding query...</span>
          </div>
        )}
      </div>
    </StepCard>
  )
}

// ---- Step 2: Document Retrieval ----
function Step2Retrieval({ question, phase, onDocClick }) {
  return (
    <StepCard
      icon={<IconSearch />}
      title="Document Retrieval"
      badge={phase > 2 ? `${question.retrievedDocs.length} documents retrieved` : 'Searching...'}
      visible={phase >= 2}
    >
      {phase >= 3 ? (
        <div className="space-y-2">
          {question.retrievedDocs.map((doc, idx) => (
            <button
              key={doc.id}
              onClick={() => onDocClick(doc)}
              className="w-full text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 hover:border-gray-600 rounded-lg px-3 py-2.5 transition-all duration-150 group"
            >
              <div className="flex items-start gap-2.5">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-600/20 text-blue-400 text-xs flex items-center justify-center font-mono mt-0.5">
                  {idx + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-gray-200 text-xs font-medium leading-snug group-hover:text-white transition-colors line-clamp-1">
                    {doc.title}
                  </p>
                  <p className="text-gray-500 text-xs mt-0.5 truncate">{doc.journal}</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <span className="text-gray-500 text-xs">BM25</span>
                    <ScoreBar score={doc.bm25Score} color="blue" />
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {question.retrievedDocs.map((doc, idx) => (
            <div
              key={doc.id}
              className="fade-in bg-gray-800/50 border border-gray-700/50 rounded-lg px-3 py-2.5"
              style={{ animationDelay: `${idx * 80}ms` }}
            >
              <div className="flex items-start gap-2.5">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-600/20 text-blue-400 text-xs flex items-center justify-center font-mono mt-0.5">
                  {idx + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <button
                    onClick={() => onDocClick(doc)}
                    className="text-left"
                  >
                    <p className="text-gray-200 text-xs font-medium leading-snug hover:text-white transition-colors line-clamp-1">
                      {doc.title}
                    </p>
                  </button>
                  <p className="text-gray-500 text-xs mt-0.5 truncate">{doc.journal}</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <span className="text-gray-500 text-xs">BM25</span>
                    <ScoreBar score={doc.bm25Score} color="blue" />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </StepCard>
  )
}

// ---- Step 3: Context Reranking ----
function Step3Reranking({ question, phase, onDocClick }) {
  // Build a lookup for doc data by id
  const docMap = {}
  question.retrievedDocs.forEach((d) => { docMap[d.id] = d })

  // For each reranked entry, compute movement: new rank (1-indexed) vs prevRank
  const reranked = question.rerankedDocs.map((r, idx) => ({
    ...r,
    newRank: idx + 1,
    doc: docMap[r.docId],
  }))

  return (
    <StepCard
      icon={<IconSort />}
      title="Context Reranking"
      badge={phase > 3 ? 'cross-encoder' : 'Reranking...'}
      visible={phase >= 3}
    >
      <div className="space-y-2">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-gray-500 text-xs">Model:</span>
          <span className="text-violet-300 text-xs bg-violet-600/15 border border-violet-500/30 px-2 py-0.5 rounded font-mono">
            cross-encoder/ms-marco-MiniLM-L-6-v2
          </span>
        </div>

        {reranked.map((item) => {
          const moved = item.newRank - item.prevRank // negative = moved up, positive = moved down
          const unchanged = moved === 0

          let moveBadge = null
          if (!unchanged) {
            const isUp = moved < 0
            moveBadge = (
              <span
                className={`flex items-center gap-0.5 text-xs font-medium ${
                  isUp ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {isUp ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
                {Math.abs(moved)}
              </span>
            )
          }

          return (
            <button
              key={item.docId}
              onClick={() => onDocClick({ ...item.doc, ceScore: item.ceScore })}
              className="w-full text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 hover:border-gray-600 rounded-lg px-3 py-2.5 transition-all duration-150 group fade-in"
            >
              <div className="flex items-start gap-2.5">
                {/* Rank indicator */}
                <div className="flex flex-col items-center gap-0.5 flex-shrink-0 pt-0.5">
                  <span className="w-5 h-5 rounded-full bg-violet-600/20 text-violet-400 text-xs flex items-center justify-center font-mono">
                    {item.newRank}
                  </span>
                  {moveBadge ? moveBadge : <span className="text-gray-600 text-xs">—</span>}
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-gray-200 text-xs font-medium leading-snug group-hover:text-white transition-colors line-clamp-1">
                    {item.doc?.title}
                  </p>
                  <p className="text-gray-500 text-xs mt-0.5 truncate">{item.doc?.journal}</p>

                  <div className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-gray-500 text-xs">BM25</span>
                      <span className="text-blue-400 text-xs font-mono">
                        {item.doc?.bm25Score.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-gray-500 text-xs">CE</span>
                      <ScoreBar score={item.ceScore} color="violet" />
                    </div>
                  </div>
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </StepCard>
  )
}

// ---- Step 4: Answer Generation ----
function Step4Generation({ question, typedAnswer, phase, configKey }) {
  const labelConfig = {
    yes: { text: 'YES', classes: 'bg-green-400/15 text-green-400 border-green-400/30' },
    maybe: { text: 'MAYBE', classes: 'bg-yellow-400/15 text-yellow-400 border-yellow-400/30' },
    no: { text: 'NO', classes: 'bg-red-400/15 text-red-400 border-red-400/30' },
  }
  const predictedLabel = question.configs[configKey].label
  const lc = labelConfig[predictedLabel]
  const isTyping = phase === 4
  const isDone = phase === 5

  return (
    <StepCard
      icon={<IconSparkles />}
      title="Answer Generation"
      badge={
        isDone
          ? 'Complete'
          : isTyping
          ? 'Generating...'
          : undefined
      }
      visible={phase >= 4}
    >
      <div className="space-y-3">
        {/* Label badge */}
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-xs">Predicted answer:</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded border ${lc.classes}`}>
            {lc.text}
          </span>
        </div>

        {/* Typed answer */}
        <div className="bg-gray-800/50 border border-gray-700/40 rounded-lg p-3">
          <p className={`text-gray-200 text-sm leading-relaxed whitespace-pre-line ${isTyping ? 'typewriter-cursor' : ''}`}>
            {typedAnswer}
          </p>
        </div>
      </div>
    </StepCard>
  )
}

// ---- Main RAGSteps Component ----
export default function RAGSteps({ question, phase, typedAnswer, onDocClick, useQR, useCR, configKey }) {
  if (!question) return null

  return (
    <div className="space-y-3 my-4">
      {/* Divider label */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-gray-800" />
        <span className="text-gray-600 text-xs font-medium uppercase tracking-wider">RAG Pipeline</span>
        <div className="flex-1 h-px bg-gray-800" />
      </div>

      {useQR && <Step1QueryRewriting question={question} phase={phase} />}
      <Step2Retrieval question={question} phase={phase} onDocClick={onDocClick} />
      {useCR && <Step3Reranking question={question} phase={phase} onDocClick={onDocClick} />}
      <Step4Generation question={question} typedAnswer={typedAnswer} phase={phase} configKey={configKey} />
    </div>
  )
}
