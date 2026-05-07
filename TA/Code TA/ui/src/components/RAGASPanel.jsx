import React, { useState } from 'react'
import { RAGAS_METRICS_INFO } from '../data/realData'

// Color utility per metric
const METRIC_COLORS = {
  faithfulness: {
    bg: 'bg-emerald-500/15',
    border: 'border-emerald-500/40',
    text: 'text-emerald-300',
    dot: 'bg-emerald-500',
    highlight: 'bg-red-500/20 border-b-2 border-red-500/60',
  },
  context_recall: {
    bg: 'bg-sky-500/15',
    border: 'border-sky-500/40',
    text: 'text-sky-300',
    dot: 'bg-sky-500',
    highlight: 'bg-red-500/20 border-b-2 border-red-500/60',
  },
  answer_relevancy: {
    bg: 'bg-violet-500/15',
    border: 'border-violet-500/40',
    text: 'text-violet-300',
    dot: 'bg-violet-500',
    highlight: 'bg-red-500/20 border-b-2 border-red-500/60',
  },
  context_precision: {
    bg: 'bg-amber-500/15',
    border: 'border-amber-500/40',
    text: 'text-amber-300',
    dot: 'bg-amber-500',
    highlight: 'bg-red-500/20 border-b-2 border-red-500/60',
  },
}

function MetricIcon({ metric }) {
  const common = 'w-4 h-4'
  if (metric === 'faithfulness') {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className={common} viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
      </svg>
    )
  }
  if (metric === 'context_recall') {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className={common} viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
      </svg>
    )
  }
  if (metric === 'answer_relevancy') {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className={common} viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
      </svg>
    )
  }
  // context_precision
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className={common} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M3 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-.293.707L12 11.414V15a1 1 0 01-.293.707l-2 2A1 1 0 018 17v-5.586L3.293 6.707A1 1 0 013 6V3z" clipRule="evenodd" />
    </svg>
  )
}

function MetricScoreBar({ metric, value, label }) {
  const colors = METRIC_COLORS[metric]
  const pct = value !== null && value !== undefined ? value * 100 : 0
  const isGood = value >= 0.8
  const isMed = value >= 0.5 && value < 0.8
  const barColor = isGood ? 'bg-green-500' : isMed ? 'bg-yellow-500' : 'bg-red-500'

  return (
    <div className={`${colors.bg} ${colors.border} border rounded-lg p-3`}>
      <div className="flex items-center justify-between mb-2">
        <div className={`flex items-center gap-1.5 ${colors.text}`}>
          <MetricIcon metric={metric} />
          <span className="text-xs font-semibold">{label}</span>
        </div>
        <span className="text-white text-sm font-mono font-bold">
          {value !== null && value !== undefined ? value.toFixed(3) : '---'}
        </span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// Highlight sentence that violates the active metric
function HighlightedAnswer({ sentences, activeMetric }) {
  if (!sentences || sentences.length === 0) {
    return <p className="text-gray-500 text-xs italic">Anotasi per-kalimat tidak tersedia untuk sampel ini.</p>
  }

  return (
    <div className="space-y-1 leading-relaxed">
      {sentences.map((sent, i) => {
        const hasViolation = sent.violations?.includes(activeMetric)
        const colors = METRIC_COLORS[activeMetric]
        return (
          <span key={i}>
            <span
              className={
                hasViolation
                  ? `${colors.highlight} px-1 py-0.5 rounded`
                  : 'text-gray-300'
              }
              title={
                hasViolation
                  ? sent[`reason_${activeMetric}`] || 'Violates this metric'
                  : ''
              }
            >
              {sent.text}
            </span>{' '}
          </span>
        )
      })}
    </div>
  )
}

function ReferenceCoverage({ sentences, activeMetric }) {
  if (activeMetric !== 'context_recall' || !sentences) return null

  return (
    <div className="space-y-1.5 mt-3">
      <p className="text-xs text-sky-300 font-semibold mb-1">
        Kalimat Reference dan Cakupan Konteks:
      </p>
      {sentences.map((sent, i) => (
        <div
          key={i}
          className={`text-xs p-2 rounded border ${
            sent.covered_by_context
              ? 'bg-green-500/10 border-green-500/30 text-green-200'
              : 'bg-red-500/15 border-red-500/40 text-red-200'
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="flex-shrink-0">
              {sent.covered_by_context ? '✓' : '✗'}
            </span>
            <div className="flex-1">
              <p>{sent.text}</p>
              {!sent.covered_by_context && sent.reason && (
                <p className="text-red-300 text-[11px] mt-1 italic">
                  {sent.reason}
                </p>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function ContextRelevanceList({ items, docs, activeMetric }) {
  if (activeMetric !== 'context_precision' || !items) return null

  return (
    <div className="space-y-1.5 mt-3">
      <p className="text-xs text-amber-300 font-semibold mb-1">
        Relevansi Tiap Konteks (by rank):
      </p>
      {items.map((item) => (
        <div
          key={item.doc_idx}
          className={`text-xs p-2 rounded border ${
            item.relevant
              ? 'bg-green-500/10 border-green-500/30 text-green-200'
              : 'bg-red-500/15 border-red-500/40 text-red-200'
          }`}
        >
          <div className="flex items-start gap-2">
            <span className="flex-shrink-0 font-mono text-[10px] bg-gray-800 px-1.5 py-0.5 rounded">
              Rank #{item.doc_idx + 1}
            </span>
            <span className="flex-shrink-0">
              {item.relevant ? '✓ relevan' : '✗ tidak relevan'}
            </span>
          </div>
          <p className="text-[11px] mt-1 opacity-90">{item.reason}</p>
        </div>
      ))}
    </div>
  )
}

export default function RAGASPanel({ sample, configKey }) {
  const [activeMetric, setActiveMetric] = useState('faithfulness')

  if (!sample) return null
  const cfg = sample.configs?.[configKey]
  if (!cfg) return null

  const annotations = sample.ragas_annotations

  const metrics = [
    { key: 'faithfulness', label: 'Faithfulness', value: cfg.faithfulness },
    { key: 'context_recall', label: 'Context Recall', value: cfg.context_recall },
    { key: 'answer_relevancy', label: 'Answer Relevancy', value: cfg.answer_relevancy },
    { key: 'context_precision', label: 'Context Precision', value: cfg.context_precision },
  ]

  const info = RAGAS_METRICS_INFO[activeMetric]
  const colors = METRIC_COLORS[activeMetric]

  return (
    <div className="mt-4 bg-gray-900/60 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-6 h-6 rounded-md bg-gradient-to-br from-violet-600 to-pink-600 flex items-center justify-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5 text-white" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M5 2a1 1 0 011 1v1h1a1 1 0 010 2H6v1a1 1 0 01-2 0V6H3a1 1 0 010-2h1V3a1 1 0 011-1zm0 10a1 1 0 011 1v1h1a1 1 0 110 2H6v1a1 1 0 11-2 0v-1H3a1 1 0 110-2h1v-1a1 1 0 011-1zM12 2a1 1 0 01.967.744L14.146 7.2 17.5 9.134a1 1 0 010 1.732l-3.354 1.935-1.18 4.455a1 1 0 01-1.933 0L9.854 12.8 6.5 10.866a1 1 0 010-1.732l3.354-1.935 1.18-4.455A1 1 0 0112 2z" clipRule="evenodd" />
          </svg>
        </div>
        <div>
          <h3 className="text-white text-sm font-semibold">
            Analisis RAGAS (4 Metrik)
          </h3>
          <p className="text-gray-500 text-[11px]">
            Klik metrik untuk melihat kalimat yang melanggar
          </p>
        </div>
      </div>

      {/* Metric tabs / score bars */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        {metrics.map((m) => (
          <button
            key={m.key}
            onClick={() => setActiveMetric(m.key)}
            className={`text-left transition-all ${
              activeMetric === m.key
                ? 'ring-2 ring-violet-500/60 rounded-lg'
                : 'hover:opacity-80'
            }`}
          >
            <MetricScoreBar metric={m.key} value={m.value} label={m.label} />
          </button>
        ))}
      </div>

      {/* Active metric explanation */}
      <div className={`${colors.bg} ${colors.border} border rounded-lg p-3 mb-3`}>
        <div className="flex items-start gap-2 mb-1.5">
          <div className={`${colors.text} mt-0.5`}>
            <MetricIcon metric={activeMetric} />
          </div>
          <div className="flex-1">
            <h4 className={`${colors.text} text-xs font-semibold`}>
              {info.name}
            </h4>
            <p className="text-gray-300 text-[11px] leading-relaxed mt-0.5">
              {info.description}
            </p>
            <p className="text-gray-400 text-[10px] italic font-mono mt-1">
              {info.formula}
            </p>
          </div>
        </div>
      </div>

      {/* Violation breakdown */}
      {annotations ? (
        <div>
          {/* Answer with highlights (for faithfulness, answer_relevancy) */}
          {(activeMetric === 'faithfulness' || activeMetric === 'answer_relevancy') && (
            <div>
              <p className="text-gray-400 text-xs font-semibold mb-1.5">
                Jawaban LLM (kalimat merah = melanggar {info.name}):
              </p>
              <div className="bg-gray-950/50 border border-gray-800 rounded-md p-3 text-xs">
                <HighlightedAnswer
                  sentences={annotations.answer_sentences}
                  activeMetric={activeMetric}
                />
              </div>

              {/* Detail violations list */}
              {(() => {
                const violations = annotations.answer_sentences?.filter((s) =>
                  s.violations?.includes(activeMetric)
                )
                if (!violations || violations.length === 0) {
                  return (
                    <p className="text-green-400 text-xs mt-2">
                      ✓ Tidak ada kalimat yang melanggar {info.name} pada sampel ini.
                    </p>
                  )
                }
                return (
                  <div className="mt-2 space-y-1.5">
                    <p className="text-red-300 text-xs font-semibold">
                      {violations.length} kalimat melanggar:
                    </p>
                    {violations.map((v, i) => (
                      <div
                        key={i}
                        className="text-[11px] p-2 rounded bg-red-500/10 border border-red-500/30"
                      >
                        <p className="text-red-200 italic mb-1">"{v.text}"</p>
                        {v[`reason_${activeMetric}`] && (
                          <p className="text-red-300">
                            <span className="font-semibold">Alasan:</span>{' '}
                            {v[`reason_${activeMetric}`]}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                )
              })()}
            </div>
          )}

          {/* Reference coverage (for context_recall) */}
          {activeMetric === 'context_recall' && (
            <ReferenceCoverage
              sentences={annotations.reference_sentences}
              activeMetric={activeMetric}
            />
          )}

          {/* Context relevance (for context_precision) */}
          {activeMetric === 'context_precision' && (
            <ContextRelevanceList
              items={annotations.contexts_relevance}
              docs={sample.retrieved_docs}
              activeMetric={activeMetric}
            />
          )}
        </div>
      ) : (
        <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-3">
          <p className="text-gray-400 text-xs">
            <span className="text-gray-300 font-semibold">Skor agregat:</span> nilai metrik
            untuk konfigurasi ini adalah{' '}
            <span className="text-white font-mono">
              {cfg[activeMetric] !== null && cfg[activeMetric] !== undefined
                ? cfg[activeMetric].toFixed(3)
                : '---'}
            </span>
            .
          </p>
          <p className="text-gray-500 text-[11px] mt-1">
            Anotasi per-kalimat hanya tersedia untuk sampel pilihan (idx 0 dan 29). Pilih
            sampel "Mitochondria Lace Plant" atau "Visceral Adipose Tissue" untuk melihat
            breakdown detail.
          </p>
        </div>
      )}
    </div>
  )
}
