import React, { useState } from 'react'

/**
 * Right-side panel: shows the RAG pipeline details for the most recent
 * assistant message — original query, rewritten query, top retrieved docs
 * with per-stage scores, and per-stage latency.
 *
 * If a message arrived from the mock service (no pipeline payload), the
 * panel shows a friendly placeholder.
 *
 * Props:
 *   pipeline  — { original_query, rewritten_query, retrieved_docs, latency_ms } | null
 *   model     — string, the LLM name
 *   isOpen    — drawer visibility (mobile)
 *   onClose() — close drawer
 */
export default function PipelinePanel({ pipeline, model, isOpen, onClose }) {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/30 xl:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed xl:static top-0 right-0 z-40 h-full w-80 flex-shrink-0
                    bg-slate-50 border-l border-slate-200
                    flex flex-col transition-transform duration-200
                    ${isOpen ? 'translate-x-0' : 'translate-x-full xl:translate-x-0'}`}
      >
        <header className="flex-shrink-0 px-4 py-3 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h2 className="text-[13px] font-semibold text-slate-900 uppercase tracking-wider">
              Pipeline details
            </h2>
            <p className="text-[11px] text-slate-500 mt-0.5">
              {model ? `Model: ${model}` : 'Hybrid + QR + CR'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="xl:hidden text-slate-500 hover:text-slate-900 p-1 rounded"
            aria-label="Close pipeline details"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5 text-[13px]">
          {!pipeline ? (
            <EmptyState />
          ) : (
            <>
              <QuerySection
                label="Original query"
                value={pipeline.original_query}
              />
              <QuerySection
                label="Rewritten query (QR)"
                value={pipeline.rewritten_query}
                highlight
              />

              <section>
                <SectionHeading>Top retrieved documents</SectionHeading>
                {pipeline.retrieved_docs?.length ? (
                  <ol className="space-y-2.5">
                    {pipeline.retrieved_docs.map((doc) => (
                      <DocCard key={`${doc.rank}-${doc.pubid}`} doc={doc} />
                    ))}
                  </ol>
                ) : (
                  <p className="text-[12px] text-slate-500">No documents retrieved.</p>
                )}
              </section>

              {pipeline.latency_ms && Object.keys(pipeline.latency_ms).length > 0 && (
                <section>
                  <SectionHeading>Latency per stage</SectionHeading>
                  <ul className="text-[12px] text-slate-600 space-y-1">
                    {Object.entries(pipeline.latency_ms).map(([k, ms]) => (
                      <li key={k} className="flex justify-between">
                        <span className="text-slate-500">{prettyStage(k)}</span>
                        <span className="font-mono tabular-nums">{ms.toFixed(0)}ms</span>
                      </li>
                    ))}
                    <li className="flex justify-between font-medium text-slate-900 pt-1 border-t border-slate-200">
                      <span>Total</span>
                      <span className="font-mono tabular-nums">
                        {Object.values(pipeline.latency_ms)
                          .reduce((a, b) => a + b, 0)
                          .toFixed(0)}
                        ms
                      </span>
                    </li>
                  </ul>
                </section>
              )}
            </>
          )}
        </div>
      </aside>
    </>
  )
}

function EmptyState() {
  return (
    <div className="text-center py-12 px-4">
      <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-slate-200 flex items-center justify-center">
        <span aria-hidden="true" className="text-xl">🔍</span>
      </div>
      <p className="text-[13px] text-slate-600 leading-relaxed">
        Ask a question to see the retrieval pipeline in action — query rewriting,
        top documents, fusion scores, and reranker results.
      </p>
    </div>
  )
}

function SectionHeading({ children }) {
  return (
    <h3 className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
      {children}
    </h3>
  )
}

function QuerySection({ label, value, highlight }) {
  return (
    <section>
      <SectionHeading>{label}</SectionHeading>
      <p
        className={`text-[13px] leading-relaxed rounded-lg px-3 py-2 border
                    ${
                      highlight
                        ? 'bg-blue-50 border-blue-200 text-blue-900'
                        : 'bg-white border-slate-200 text-slate-800'
                    }`}
      >
        {value || '—'}
      </p>
    </section>
  )
}

function DocCard({ doc }) {
  const [expanded, setExpanded] = useState(false)
  const snippet = doc.text || ''
  const cap = 200
  const long = snippet.length > cap
  const shown = expanded || !long ? snippet : snippet.slice(0, cap).trimEnd() + '…'

  return (
    <li className="bg-white border border-slate-200 rounded-lg p-3">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="flex-shrink-0 w-5 h-5 rounded bg-slate-100 text-slate-700 text-[11px] font-semibold flex items-center justify-center">
            {doc.rank}
          </span>
          {doc.pubid && (
            <a
              href={`https://pubmed.ncbi.nlm.nih.gov/${doc.pubid}/`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[12px] text-blue-600 hover:underline truncate"
            >
              PMID {doc.pubid}
            </a>
          )}
          {doc.section_label && (
            <span className="text-[10.5px] uppercase tracking-wider text-slate-500 flex-shrink-0">
              {doc.section_label}
            </span>
          )}
        </div>
      </div>

      <p className="text-[12.5px] text-slate-700 leading-relaxed whitespace-pre-wrap">
        {shown}
        {long && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="ml-1 text-blue-600 hover:underline text-[12px]"
          >
            {expanded ? 'show less' : 'show more'}
          </button>
        )}
      </p>

      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
        {fmtScore('BM25', doc.bm25_score, 2)}
        {fmtScore('Dense', doc.dense_score, 3)}
        {fmtScore('RRF', doc.rrf_score, 4)}
        {fmtScore('Rerank', doc.rerank_score, 3)}
      </div>
    </li>
  )
}

function fmtScore(label, value, digits) {
  if (value === null || value === undefined) return null
  return (
    <span key={label} className="flex justify-between">
      <span>{label}</span>
      <span className="font-mono tabular-nums text-slate-700">{Number(value).toFixed(digits)}</span>
    </span>
  )
}

function prettyStage(key) {
  return key
    .replace(/_ms$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase())
}
