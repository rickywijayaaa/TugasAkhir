import React, { useState, useEffect } from 'react'
import { CONFIGS } from '../data/realData'

// ---- Icons ----
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
function IconMerge() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M13.707 5.293a1 1 0 010 1.414L9.414 11l4.293 4.293a1 1 0 01-1.414 1.414l-5-5a1 1 0 010-1.414l5-5a1 1 0 011.414 0z" clipRule="evenodd" />
      <path fillRule="evenodd" d="M18.707 5.293a1 1 0 010 1.414L14.414 11l4.293 4.293a1 1 0 01-1.414 1.414l-5-5a1 1 0 010-1.414l5-5a1 1 0 011.414 0z" clipRule="evenodd" />
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

function StepCard({ icon, title, badge, children, visible, accent = 'violet' }) {
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

  const accentMap = {
    violet: 'bg-violet-600/20 text-violet-400',
    blue: 'bg-blue-600/20 text-blue-400',
    amber: 'bg-amber-600/20 text-amber-400',
    teal: 'bg-teal-600/20 text-teal-400',
    orange: 'bg-orange-600/20 text-orange-400',
  }

  return (
    <div
      className={`bg-gray-900 border border-gray-700/60 rounded-xl overflow-hidden transition-all duration-300 ${
        mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
      }`}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700/40">
        <div className="flex items-center gap-2.5">
          <div
            className={`w-6 h-6 rounded-md flex items-center justify-center ${accentMap[accent]}`}
          >
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
      <div className="p-4">{children}</div>
    </div>
  )
}

// ---- Step: Query Rewriting ----
function StepQR({ question, phase }) {
  return (
    <StepCard
      icon={<IconPencil />}
      title="Query Rewriting"
      badge={phase > 1 ? 'Completed' : 'Processing...'}
      visible={phase >= 1}
      accent="violet"
    >
      <div className="space-y-3">
        <div>
          <p className="text-gray-500 text-xs font-medium mb-1.5 uppercase tracking-wide">
            Original Query
          </p>
          <p className="text-gray-500 text-sm line-through leading-relaxed">
            {question.question}
          </p>
        </div>
        {phase >= 2 && question.rewritten_query && (
          <div className="fade-in">
            <div className="flex items-center gap-2 mb-1.5">
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wide">
                Rewritten Query
              </p>
              <span className="text-xs text-violet-400 bg-violet-400/10 px-1.5 py-0.5 rounded">
                expanded
              </span>
            </div>
            <p className="text-violet-300 text-sm leading-relaxed border-l-2 border-violet-500/50 pl-3">
              {question.rewritten_query}
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

// ---- Step: BM25-only Retrieval ----
function StepBM25({ question, phase, onDocClick }) {
  return (
    <StepCard
      icon={<IconSearch />}
      title="BM25 Retrieval"
      badge={phase > 2 ? `${question.retrieved_docs.length} dokumen` : 'Searching...'}
      visible={phase >= 2}
      accent="blue"
    >
      {phase >= 3 ? (
        <div className="space-y-2">
          <p className="text-gray-500 text-xs mb-2">
            BM25 (sparse/keyword matching) mengambil {question.retrieved_docs.length} dokumen
            teratas
          </p>
          {question.retrieved_docs.map((doc, idx) => (
            <button
              key={doc.id}
              onClick={() => onDocClick(doc)}
              className="w-full text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 hover:border-gray-600 rounded-lg px-3 py-2 transition-all"
            >
              <div className="flex items-center gap-2.5">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-600/20 text-blue-400 text-xs flex items-center justify-center font-mono">
                  {idx + 1}
                </span>
                <p className="flex-1 text-gray-300 text-xs leading-snug line-clamp-1">
                  {doc.content.slice(0, 100)}...
                </p>
                {doc.bm25_score !== null && doc.bm25_score !== undefined && (
                  <span className="flex-shrink-0 text-xs font-mono text-blue-400">
                    BM25: {doc.bm25_score.toFixed(2)}
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <span className="text-gray-500 text-xs">
            Scanning 1706 dokumen PubMedQA...
          </span>
        </div>
      )}
    </StepCard>
  )
}

// ---- Step: Hybrid Retrieval (BM25 + Dense + RRF) ----
function StepHybrid({ question, phase, onDocClick }) {
  return (
    <StepCard
      icon={<IconMerge />}
      title="Hybrid Retrieval (BM25 + Dense via RRF)"
      badge={phase > 2 ? `top-${question.retrieved_docs.length} fused` : 'Fusing...'}
      visible={phase >= 2}
      accent="amber"
    >
      {phase >= 3 ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-blue-500/10 border border-blue-500/30 rounded p-2">
              <p className="text-blue-400 text-[10px] uppercase font-semibold mb-0.5">
                Sparse: BM25
              </p>
              <p className="text-gray-300">Keyword matching · top-50</p>
            </div>
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded p-2">
              <p className="text-emerald-400 text-[10px] uppercase font-semibold mb-0.5">
                Dense: OpenAI Embed
              </p>
              <p className="text-gray-300">Semantic · text-embed-3-small · top-50</p>
            </div>
          </div>
          <div className="bg-amber-500/10 border border-amber-500/30 rounded p-2 text-xs">
            <p className="text-amber-400 text-[10px] uppercase font-semibold mb-1">
              Reciprocal Rank Fusion (k=60)
            </p>
            <p className="text-gray-300 font-mono text-[11px]">
              RRF(d) = Σ 1 / (k + rank_r(d))
            </p>
          </div>

          <div className="space-y-1.5">
            <p className="text-gray-500 text-xs mb-1">
              Top {question.retrieved_docs.length} dokumen setelah RRF fusion:
            </p>
            {question.retrieved_docs.map((doc, idx) => (
              <button
                key={doc.id}
                onClick={() => onDocClick(doc)}
                className="w-full text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 rounded-lg px-3 py-2"
              >
                <div className="flex items-center gap-2.5">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-amber-600/20 text-amber-400 text-xs flex items-center justify-center font-mono">
                    {idx + 1}
                  </span>
                  <p className="flex-1 text-gray-300 text-xs leading-snug line-clamp-1">
                    {doc.content.slice(0, 90)}...
                  </p>
                  <div className="flex items-center gap-1.5 flex-shrink-0 text-[10px] font-mono">
                    {doc.bm25_score !== null && doc.bm25_score !== undefined && (
                      <span className="text-blue-400">B:{doc.bm25_score.toFixed(1)}</span>
                    )}
                    {doc.dense_score !== null && doc.dense_score !== undefined && (
                      <span className="text-emerald-400">D:{doc.dense_score.toFixed(2)}</span>
                    )}
                    {doc.rrf_score !== null && doc.rrf_score !== undefined && (
                      <span className="text-amber-400">R:{doc.rrf_score.toFixed(4)}</span>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
          <span className="text-gray-500 text-xs">
            Running BM25 + Dense in parallel, applying RRF...
          </span>
        </div>
      )}
    </StepCard>
  )
}

// ---- Step: CrossEncoder Reranking ----
function StepCR({ question, phase, onDocClick }) {
  return (
    <StepCard
      icon={<IconSort />}
      title="Context Reranking (CrossEncoder)"
      badge={phase > 3 ? `top-${question.retrieved_docs.length} reranked` : 'Reranking...'}
      visible={phase >= 3}
      accent="teal"
    >
      {phase >= 4 ? (
        <div className="space-y-2">
          <p className="text-gray-500 text-xs mb-2">
            ms-marco-MiniLM-L-6-v2 menilai setiap (query, doc) pair dan mengurutkan ulang:
          </p>
          {question.retrieved_docs.map((doc, idx) => (
            <button
              key={doc.id}
              onClick={() => onDocClick(doc)}
              className="w-full text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-2.5">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-teal-600/20 text-teal-400 text-xs flex items-center justify-center font-mono">
                  {idx + 1}
                </span>
                <p className="flex-1 text-gray-300 text-xs leading-snug line-clamp-1">
                  {doc.content.slice(0, 90)}...
                </p>
                {doc.reranker_score !== null && doc.reranker_score !== undefined && (
                  <span className="flex-shrink-0 text-xs font-mono text-teal-400">
                    CE: {doc.reranker_score.toFixed(3)}
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
          <span className="text-gray-500 text-xs">
            CrossEncoder scoring (query, doc) pairs...
          </span>
        </div>
      )}
    </StepCard>
  )
}

// ---- Step: LLM Generation ----
function StepGeneration({ typedAnswer, phase }) {
  return (
    <StepCard
      icon={<IconSparkles />}
      title="LLM Generation"
      badge={phase === 5 ? 'Completed' : 'Generating...'}
      visible={phase >= 4}
      accent="orange"
    >
      <div className="space-y-2">
        {phase === 4 && (
          <div>
            <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">
              {typedAnswer}
              <span className="inline-block w-1 h-4 bg-violet-400 ml-0.5 animate-pulse align-middle" />
            </p>
          </div>
        )}
        {phase === 5 && (
          <p className="text-emerald-400 text-xs">
            ✓ Jawaban lengkap dihasilkan. Lihat breakdown RAGAS di panel bawah.
          </p>
        )}
      </div>
    </StepCard>
  )
}

// ---- Main component ----
export default function RAGSteps({
  question,
  phase,
  typedAnswer,
  onDocClick,
  selectedConfig,
}) {
  const config = CONFIGS.find((c) => c.key === selectedConfig) || CONFIGS[3]

  const showQR = config.useQR
  const showHybrid = config.useHybrid
  const showBM25only = !config.useHybrid
  const showCR = config.useCrossEncoder

  return (
    <div className="space-y-3 mt-2">
      {showQR && <StepQR question={question} phase={phase} />}
      {showHybrid ? (
        <StepHybrid question={question} phase={phase} onDocClick={onDocClick} />
      ) : (
        showBM25only && <StepBM25 question={question} phase={phase} onDocClick={onDocClick} />
      )}
      {showCR && <StepCR question={question} phase={phase} onDocClick={onDocClick} />}
      <StepGeneration typedAnswer={typedAnswer} phase={phase} />
    </div>
  )
}
