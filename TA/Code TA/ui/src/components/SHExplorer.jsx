import React, { useState, useMemo, useEffect } from 'react'
import shData from '../data/sh_baseline_500.json'

const labelStyles = {
  yes:   { bg: 'bg-green-500/15',  text: 'text-green-300',  border: 'border-green-500/40' },
  no:    { bg: 'bg-red-500/15',    text: 'text-red-300',    border: 'border-red-500/40' },
  maybe: { bg: 'bg-yellow-500/15', text: 'text-yellow-300', border: 'border-yellow-500/40' },
}

const sectionColors = {
  BACKGROUND: 'text-slate-400',
  OBJECTIVE: 'text-cyan-400',
  OBJECTIVES: 'text-cyan-400',
  AIMS: 'text-cyan-400',
  PURPOSE: 'text-cyan-400',
  METHODS: 'text-violet-400',
  METHOD: 'text-violet-400',
  'MATERIALS AND METHODS': 'text-violet-400',
  'MATERIAL AND METHODS': 'text-violet-400',
  'PATIENTS AND METHODS': 'text-violet-400',
  RESULTS: 'text-amber-400',
  CONCLUSION: 'text-emerald-400',
  CONCLUSIONS: 'text-emerald-400',
}

function LabelBadge({ label, small = false }) {
  const s = labelStyles[label] || labelStyles.maybe
  return (
    <span
      className={`inline-flex items-center font-bold uppercase tracking-wider border rounded ${s.bg} ${s.text} ${s.border} ${
        small ? 'text-[9px] px-1.5 py-0.5' : 'text-xs px-2 py-1'
      }`}
    >
      {label}
    </span>
  )
}

function CorrectMark({ ok, small = false }) {
  return (
    <span
      className={`inline-flex items-center justify-center rounded ${
        ok ? 'bg-green-500/20 text-green-400 border-green-500/40' : 'bg-red-500/20 text-red-400 border-red-500/40'
      } border ${small ? 'text-[9px] w-4 h-4' : 'text-xs w-5 h-5'} font-bold`}
      title={ok ? 'Benar' : 'Salah'}
    >
      {ok ? '✓' : '✗'}
    </span>
  )
}

export default function SHExplorer({ onClose }) {
  const { stats, samples } = shData

  const [selectedIdx, setSelectedIdx] = useState(samples[0].idx)
  const [search, setSearch] = useState('')
  const [filterCorrect, setFilterCorrect] = useState('all') // all | correct | wrong
  const [filterLabel, setFilterLabel] = useState('all') // all | yes | no | maybe
  const [filterPredMatch, setFilterPredMatch] = useState('all') // all | match | mismatch

  // Filtered list
  const filtered = useMemo(() => {
    return samples.filter((s) => {
      if (filterCorrect === 'correct' && !s.is_correct) return false
      if (filterCorrect === 'wrong' && s.is_correct) return false
      if (filterLabel !== 'all' && s.ground_truth !== filterLabel) return false
      if (filterPredMatch === 'match' && s.predicted_label !== s.ground_truth) return false
      if (filterPredMatch === 'mismatch' && s.predicted_label === s.ground_truth) return false
      if (search.trim()) {
        const q = search.toLowerCase()
        if (
          !s.question.toLowerCase().includes(q) &&
          !String(s.idx).includes(q) &&
          !String(s.pubid).includes(q)
        )
          return false
      }
      return true
    })
  }, [samples, search, filterCorrect, filterLabel, filterPredMatch])

  // Auto-select first filtered if current selection not in filtered
  useEffect(() => {
    if (!filtered.find((s) => s.idx === selectedIdx) && filtered.length > 0) {
      setSelectedIdx(filtered[0].idx)
    }
  }, [filtered, selectedIdx])

  const selected = samples.find((s) => s.idx === selectedIdx)

  // ESC to close
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  // Parse answer to separate reasoning from final label
  const parseAnswer = (answer) => {
    if (!answer) return { reasoning: '', finalLabel: '' }
    const lines = answer.trim().split('\n').filter((l) => l.trim())
    const lastLine = lines[lines.length - 1]?.trim().toLowerCase()
    if (['yes', 'no', 'maybe'].includes(lastLine)) {
      return {
        reasoning: lines.slice(0, -1).join('\n').trim(),
        finalLabel: lastLine,
      }
    }
    return { reasoning: answer, finalLabel: '' }
  }

  const { reasoning, finalLabel } = parseAnswer(selected?.answer || '')

  return (
    <div className="fixed inset-0 bg-black/85 z-50 flex flex-col" onClick={onClose}>
      <div
        className="w-full h-full flex flex-col bg-gray-950"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900">
          <div>
            <h2 className="text-white font-semibold text-lg flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />
              SH Baseline Explorer
              <span className="text-xs font-normal text-gray-500 ml-2">
                {stats.total} sampel · {stats.config} · {stats.model}
              </span>
            </h2>
            <p className="text-gray-500 text-xs mt-0.5">
              Akurasi: <span className="text-emerald-400 font-mono font-bold">{(stats.accuracy * 100).toFixed(2)}%</span>
              {' '}({stats.correct}/{stats.total}) ·
              <span className="text-green-400 font-mono ml-2">yes {(stats.per_class.yes.accuracy * 100).toFixed(1)}%</span> ·
              <span className="text-red-400 font-mono ml-2">no {(stats.per_class.no.accuracy * 100).toFixed(1)}%</span> ·
              <span className="text-yellow-400 font-mono ml-2">maybe {(stats.per_class.maybe.accuracy * 100).toFixed(1)}%</span>
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl leading-none px-3 py-1 hover:bg-gray-800 rounded"
            title="Tutup (ESC)"
          >
            ×
          </button>
        </div>

        <div className="flex-1 flex min-h-0">
          {/* LEFT: List */}
          <div className="w-96 flex-shrink-0 flex flex-col border-r border-gray-800 bg-gray-900">
            {/* Filters */}
            <div className="p-3 border-b border-gray-800 space-y-2">
              <input
                type="text"
                placeholder="Cari pertanyaan / idx / pubid…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-gray-800 text-gray-200 placeholder-gray-500 border border-gray-700 rounded-md px-2.5 py-1.5 text-xs focus:outline-none focus:border-violet-500"
              />
              <div className="grid grid-cols-3 gap-1.5 text-[10px]">
                <select
                  value={filterCorrect}
                  onChange={(e) => setFilterCorrect(e.target.value)}
                  className="bg-gray-800 text-gray-300 border border-gray-700 rounded px-1.5 py-1 focus:outline-none focus:border-violet-500"
                >
                  <option value="all">Status: Semua</option>
                  <option value="correct">Benar</option>
                  <option value="wrong">Salah</option>
                </select>
                <select
                  value={filterLabel}
                  onChange={(e) => setFilterLabel(e.target.value)}
                  className="bg-gray-800 text-gray-300 border border-gray-700 rounded px-1.5 py-1 focus:outline-none focus:border-violet-500"
                >
                  <option value="all">GT: Semua</option>
                  <option value="yes">GT: yes</option>
                  <option value="no">GT: no</option>
                  <option value="maybe">GT: maybe</option>
                </select>
                <select
                  value={filterPredMatch}
                  onChange={(e) => setFilterPredMatch(e.target.value)}
                  className="bg-gray-800 text-gray-300 border border-gray-700 rounded px-1.5 py-1 focus:outline-none focus:border-violet-500"
                >
                  <option value="all">Pred: Semua</option>
                  <option value="match">Match GT</option>
                  <option value="mismatch">Mismatch GT</option>
                </select>
              </div>
              <p className="text-gray-500 text-[10px]">
                {filtered.length} dari {samples.length} sampel
              </p>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto">
              {filtered.length === 0 ? (
                <div className="p-6 text-center text-gray-500 text-sm">
                  Tidak ada sampel cocok filter.
                </div>
              ) : (
                <div className="divide-y divide-gray-800">
                  {filtered.map((s) => {
                    const isActive = s.idx === selectedIdx
                    return (
                      <button
                        key={s.idx}
                        onClick={() => setSelectedIdx(s.idx)}
                        className={`w-full text-left px-3 py-2.5 transition-colors group ${
                          isActive ? 'bg-violet-600/15 border-l-2 border-violet-500' : 'hover:bg-gray-800/50 border-l-2 border-transparent'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-gray-500 text-[10px] font-mono">#{s.idx}</span>
                          <span className="text-gray-600 text-[10px] font-mono">·{s.pubid}</span>
                          <div className="ml-auto flex items-center gap-1">
                            <LabelBadge label={s.ground_truth} small />
                            <span className="text-gray-600 text-[10px]">→</span>
                            <LabelBadge label={s.predicted_label} small />
                            <CorrectMark ok={s.is_correct} small />
                          </div>
                        </div>
                        <p
                          className={`text-xs leading-snug line-clamp-2 ${
                            isActive ? 'text-violet-200' : 'text-gray-300 group-hover:text-gray-200'
                          }`}
                        >
                          {s.question}
                        </p>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Detail */}
          <div className="flex-1 overflow-y-auto bg-gray-950">
            {selected ? (
              <div className="max-w-4xl mx-auto p-6 space-y-5">
                {/* Question header */}
                <div className="border-b border-gray-800 pb-4">
                  <div className="flex items-center gap-2 mb-2 text-xs text-gray-500">
                    <span className="font-mono">idx #{selected.idx}</span>
                    <span>·</span>
                    <a
                      href={`https://pubmed.ncbi.nlm.nih.gov/${selected.pubid}/`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-mono text-violet-400 hover:text-violet-300 hover:underline"
                    >
                      PMID: {selected.pubid}
                    </a>
                  </div>
                  <h3 className="text-white text-lg font-semibold leading-snug mb-3">
                    {selected.question}
                  </h3>
                  <div className="flex items-center gap-3 flex-wrap">
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Ground Truth:</span>
                      <LabelBadge label={selected.ground_truth} />
                    </div>
                    <span className="text-gray-700">→</span>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-500 text-xs uppercase tracking-wider font-semibold">Prediksi:</span>
                      <LabelBadge label={selected.predicted_label} />
                      <CorrectMark ok={selected.is_correct} />
                    </div>
                  </div>
                </div>

                {/* Answer */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="text-gray-300 text-xs uppercase tracking-wider font-semibold">
                      Jawaban LLM
                    </h4>
                    <span className="text-gray-600 text-[10px]">GPT-4.1-mini · temp=0</span>
                  </div>
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                    <p className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap">
                      {reasoning}
                    </p>
                    {finalLabel && (
                      <div className="mt-3 pt-3 border-t border-gray-800 flex items-center gap-2">
                        <span className="text-gray-500 text-[10px] uppercase tracking-wider font-semibold">
                          Final label
                        </span>
                        <LabelBadge label={finalLabel} />
                      </div>
                    )}
                  </div>
                </div>

                {/* Reference (ground truth long answer) */}
                <div>
                  <h4 className="text-gray-300 text-xs uppercase tracking-wider font-semibold mb-2">
                    Reference (Long Answer dari Paper)
                  </h4>
                  <div className="bg-emerald-950/30 border border-emerald-900/40 rounded-lg p-4">
                    <p className="text-emerald-100/80 text-sm leading-relaxed italic">
                      {selected.reference}
                    </p>
                  </div>
                </div>

                {/* Retrieved contexts */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-gray-300 text-xs uppercase tracking-wider font-semibold">
                      Top-{selected.contexts.length} Retrieved Contexts
                    </h4>
                    <span className="text-gray-600 text-[10px]">BM25 SH + Chroma SH + RRF</span>
                  </div>
                  <div className="space-y-2">
                    {selected.contexts.map((ctx, i) => {
                      const pid = selected.context_pubids[i]
                      const section = selected.context_sections[i]
                      const isOwn = pid === selected.pubid
                      const bm25 = selected.retrieval_scores[i]
                      const dense = selected.dense_scores[i]
                      const rrf = selected.rrf_scores[i]
                      const sectionColor = sectionColors[section?.toUpperCase()] || 'text-gray-500'

                      return (
                        <div
                          key={i}
                          className={`rounded-lg border p-3 ${
                            isOwn ? 'bg-violet-950/30 border-violet-800/50' : 'bg-gray-900 border-gray-800'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
                            <div className="flex items-center gap-2 text-[11px]">
                              <span className="text-gray-500 font-mono">[{i + 1}]</span>
                              <span className={`uppercase font-semibold tracking-wider ${sectionColor}`}>
                                {section || '—'}
                              </span>
                              <a
                                href={`https://pubmed.ncbi.nlm.nih.gov/${pid}/`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-500 font-mono hover:text-violet-400 hover:underline"
                              >
                                PMID:{pid}
                              </a>
                              {isOwn && (
                                <span className="bg-violet-500/20 text-violet-300 border border-violet-500/40 rounded text-[9px] px-1.5 py-0.5 font-bold uppercase tracking-wider">
                                  Source Paper
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500">
                              <span title="BM25 score">BM25: <span className="text-cyan-400">{bm25?.toFixed(2)}</span></span>
                              <span title="Dense (cosine sim)">D: <span className="text-amber-400">{dense?.toFixed(3)}</span></span>
                              <span title="RRF fused score">RRF: <span className="text-emerald-400">{rrf?.toFixed(3)}</span></span>
                            </div>
                          </div>
                          <p className="text-gray-300 text-xs leading-relaxed">
                            {ctx}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                Pilih sampel di kiri
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
