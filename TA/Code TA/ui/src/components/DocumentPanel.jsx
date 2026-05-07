import React, { useState, useEffect } from 'react'
import { AGGREGATE_STATS, CONFIGS, PER_LABEL_ACCURACY } from '../data/realData'

function AnimatedBar({ value, maxValue = 100, colorClass }) {
  const [width, setWidth] = useState(0)
  const pct = (value / maxValue) * 100

  useEffect(() => {
    const t = setTimeout(() => setWidth(pct), 150)
    return () => clearTimeout(t)
  }, [pct])

  return (
    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
      <div
        className={`h-full ${colorClass} rounded-full score-bar`}
        style={{ width: `${width}%` }}
      />
    </div>
  )
}

export default function DocumentPanel({ question, phase, onDocClick, selectedConfig }) {
  const [activeTab, setActiveTab] = useState('documents')
  const config = CONFIGS.find((c) => c.key === selectedConfig) || CONFIGS[3]

  useEffect(() => {
    setActiveTab('documents')
  }, [question?.idx])

  const hasRetrieval = phase >= 2
  const hasReranking = phase >= 3 && config.useCrossEncoder

  const stats = AGGREGATE_STATS[selectedConfig]
  const perLabel = PER_LABEL_ACCURACY[selectedConfig]

  // Delta vs baseline OpenAI
  const deltaAcc =
    stats.accuracy - AGGREGATE_STATS.baseline_openai.accuracy
  const deltaFaith =
    stats.faithfulness - AGGREGATE_STATS.baseline_openai.faithfulness

  return (
    <div className="w-80 flex-shrink-0 bg-gray-900 border-l border-gray-800 flex flex-col h-full overflow-hidden">
      {/* Tabs */}
      <div className="flex-shrink-0 flex border-b border-gray-800">
        <button
          onClick={() => setActiveTab('documents')}
          className={`flex-1 px-3 py-2.5 text-xs font-semibold transition-all ${
            activeTab === 'documents'
              ? 'text-violet-400 border-b-2 border-violet-500'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          Dokumen
        </button>
        <button
          onClick={() => setActiveTab('stats')}
          className={`flex-1 px-3 py-2.5 text-xs font-semibold transition-all ${
            activeTab === 'stats'
              ? 'text-violet-400 border-b-2 border-violet-500'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          Statistik
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'documents' && (
          <div className="p-4">
            {!question && (
              <div className="text-center py-8">
                <p className="text-gray-500 text-xs">
                  Pilih pertanyaan untuk melihat dokumen yang di-retrieve
                </p>
              </div>
            )}
            {question && !hasRetrieval && (
              <div className="text-center py-8">
                <p className="text-gray-500 text-xs">Menunggu retrieval...</p>
              </div>
            )}
            {question && hasRetrieval && (
              <div className="space-y-2.5">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider">
                    Top-{question.retrieved_docs.length} Konteks
                  </p>
                  {hasReranking && (
                    <span className="text-[10px] bg-teal-500/20 text-teal-300 border border-teal-500/30 px-1.5 py-0.5 rounded">
                      after CR
                    </span>
                  )}
                </div>

                {question.retrieved_docs.map((doc, idx) => (
                  <button
                    key={doc.id}
                    onClick={() => onDocClick(doc)}
                    className="w-full text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700/50 rounded-lg p-3 transition-all"
                  >
                    <div className="flex items-start gap-2 mb-2">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-violet-600/20 text-violet-400 text-[10px] flex items-center justify-center font-mono font-bold">
                        {idx + 1}
                      </span>
                      <p className="text-gray-300 text-xs leading-snug line-clamp-3">
                        {doc.content.slice(0, 180)}...
                      </p>
                    </div>
                    <div className="flex items-center justify-between gap-2 pl-7 text-[10px] font-mono">
                      {doc.bm25_score !== null && doc.bm25_score !== undefined && (
                        <span className="text-blue-400">
                          BM25: {doc.bm25_score.toFixed(2)}
                        </span>
                      )}
                      {doc.dense_score !== null && doc.dense_score !== undefined && (
                        <span className="text-emerald-400">
                          Dense: {doc.dense_score.toFixed(3)}
                        </span>
                      )}
                      {doc.rrf_score !== null && doc.rrf_score !== undefined && (
                        <span className="text-amber-400">
                          RRF: {doc.rrf_score.toFixed(4)}
                        </span>
                      )}
                      {doc.reranker_score !== null && doc.reranker_score !== undefined && (
                        <span className="text-teal-400">
                          CE: {doc.reranker_score.toFixed(3)}
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="p-4 space-y-5">
            {/* Config info */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: config.color }}
                />
                <p className="text-white text-sm font-semibold">{config.label}</p>
              </div>
              <p className="text-gray-500 text-xs">{config.model}</p>
              {config.techniques.length > 0 && (
                <div className="flex gap-1 mt-2 flex-wrap">
                  {config.techniques.map((t) => (
                    <span
                      key={t}
                      className="text-[10px] bg-violet-500/10 text-violet-300 border border-violet-500/20 px-1.5 py-0.5 rounded"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Accuracy */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <p className="text-gray-400 text-xs font-semibold">Label Accuracy</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-white text-lg font-bold">{stats.accuracy}%</span>
                  <span
                    className={`text-[10px] font-mono ${
                      deltaAcc > 0
                        ? 'text-green-400'
                        : deltaAcc < 0
                        ? 'text-red-400'
                        : 'text-gray-500'
                    }`}
                  >
                    {deltaAcc > 0 ? '+' : ''}
                    {deltaAcc.toFixed(1)}% vs BL
                  </span>
                </div>
              </div>
              <AnimatedBar value={stats.accuracy} maxValue={70} colorClass="bg-gradient-to-r from-violet-500 to-pink-500" />
              <p className="text-gray-500 text-[10px] mt-1 font-mono">
                {stats.correctCount}/500 benar
              </p>
            </div>

            {/* RAGAS metrics */}
            <div className="space-y-2">
              <p className="text-gray-400 text-xs font-semibold">4 Metrik RAGAS</p>

              <div>
                <div className="flex items-center justify-between text-xs mb-0.5">
                  <span className="text-emerald-400">Faithfulness</span>
                  <span className="text-white font-mono font-bold">
                    {stats.faithfulness.toFixed(4)}
                  </span>
                </div>
                <AnimatedBar
                  value={stats.faithfulness * 100}
                  maxValue={100}
                  colorClass="bg-emerald-500"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-xs mb-0.5">
                  <span className="text-sky-400">Context Recall</span>
                  <span className="text-white font-mono font-bold">
                    {stats.contextRecall.toFixed(4)}
                  </span>
                </div>
                <AnimatedBar
                  value={stats.contextRecall * 100}
                  maxValue={100}
                  colorClass="bg-sky-500"
                />
              </div>

              {stats.answerRelevancy !== null && (
                <div>
                  <div className="flex items-center justify-between text-xs mb-0.5">
                    <span className="text-violet-400">Answer Relevancy</span>
                    <span className="text-white font-mono font-bold">
                      {stats.answerRelevancy.toFixed(4)}
                    </span>
                  </div>
                  <AnimatedBar
                    value={stats.answerRelevancy * 100}
                    maxValue={100}
                    colorClass="bg-violet-500"
                  />
                </div>
              )}

              {stats.contextPrecision !== null && (
                <div>
                  <div className="flex items-center justify-between text-xs mb-0.5">
                    <span className="text-amber-400">Context Precision</span>
                    <span className="text-white font-mono font-bold">
                      {stats.contextPrecision.toFixed(4)}
                    </span>
                  </div>
                  <AnimatedBar
                    value={stats.contextPrecision * 100}
                    maxValue={100}
                    colorClass="bg-amber-500"
                  />
                </div>
              )}

              {(stats.answerRelevancy === null || stats.contextPrecision === null) && (
                <p className="text-gray-500 text-[10px] italic mt-1">
                  Metrik AR dan CP tidak dihitung pada evaluasi Llama awal.
                </p>
              )}
            </div>

            {/* Per-label */}
            <div>
              <p className="text-gray-400 text-xs font-semibold mb-2">
                Akurasi Per-Label
              </p>
              <div className="space-y-1.5">
                {[
                  { lbl: 'yes', color: 'bg-green-500', textColor: 'text-green-400' },
                  { lbl: 'no', color: 'bg-red-500', textColor: 'text-red-400' },
                  { lbl: 'maybe', color: 'bg-yellow-500', textColor: 'text-yellow-400' },
                ].map((c) => {
                  const d = perLabel[c.lbl]
                  const acc = (d.correct / d.total) * 100
                  return (
                    <div key={c.lbl}>
                      <div className="flex items-center justify-between text-[11px] mb-0.5">
                        <span className={c.textColor}>{c.lbl}</span>
                        <span className="text-gray-300 font-mono">
                          {d.correct}/{d.total} ({acc.toFixed(0)}%)
                        </span>
                      </div>
                      <AnimatedBar
                        value={acc}
                        maxValue={100}
                        colorClass={c.color}
                      />
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Improvement summary */}
            <div className="bg-gradient-to-br from-violet-900/30 to-pink-900/30 border border-violet-500/30 rounded-lg p-3">
              <p className="text-violet-300 text-[11px] font-semibold mb-1 uppercase tracking-wide">
                Selisih dari Baseline
              </p>
              <div className="space-y-0.5 text-[11px] font-mono">
                <div className="flex justify-between">
                  <span className="text-gray-400">vs BL Llama:</span>
                  <span
                    className={
                      stats.accuracy - AGGREGATE_STATS.baseline_llama.accuracy > 0
                        ? 'text-green-400'
                        : 'text-red-400'
                    }
                  >
                    {stats.accuracy - AGGREGATE_STATS.baseline_llama.accuracy > 0 ? '+' : ''}
                    {(stats.accuracy - AGGREGATE_STATS.baseline_llama.accuracy).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">vs BL OpenAI:</span>
                  <span
                    className={
                      deltaAcc > 0 ? 'text-green-400' : deltaAcc < 0 ? 'text-red-400' : 'text-gray-500'
                    }
                  >
                    {deltaAcc > 0 ? '+' : ''}
                    {deltaAcc.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Faithful. vs BL OA:</span>
                  <span
                    className={
                      deltaFaith > 0 ? 'text-green-400' : deltaFaith < 0 ? 'text-red-400' : 'text-gray-500'
                    }
                  >
                    {deltaFaith > 0 ? '+' : ''}
                    {deltaFaith.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
