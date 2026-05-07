import React from 'react'
import { AGGREGATE_STATS, CONFIGS, PER_LABEL_ACCURACY, GROUND_TRUTH_DIST } from '../data/realData'

// Modal component that shows full comparison across 9 configs
export default function StatsDashboard({ onClose, selectedConfig }) {
  const baseline = AGGREGATE_STATS.baseline_openai

  // Sort by accuracy desc
  const sorted = [...CONFIGS].sort(
    (a, b) => AGGREGATE_STATS[b.key].accuracy - AGGREGATE_STATS[a.key].accuracy
  )

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-800 rounded-xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 sticky top-0 bg-gray-900 z-10">
          <div>
            <h2 className="text-white text-lg font-semibold">
              Perbandingan 9 Konfigurasi RAG
            </h2>
            <p className="text-gray-400 text-xs mt-0.5">
              PubMedQA 500 sampel · Baseline Llama 3.2 vs 6 konfigurasi GPT-4.1-mini
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 flex items-center justify-center"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Best config callout */}
          <div className="p-4 bg-gradient-to-r from-orange-900/30 to-yellow-900/30 border border-orange-500/30 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">🏆</span>
              <h3 className="text-white font-semibold text-sm">
                Konfigurasi Terbaik: Hybrid + CR OpenAI
              </h3>
            </div>
            <p className="text-gray-300 text-xs leading-relaxed">
              Mencapai <span className="text-orange-300 font-bold">69,8% accuracy</span> (349/500 benar),
              meningkat <span className="text-green-400 font-bold">+14,2%</span> dari Baseline Llama
              dan <span className="text-green-400 font-bold">+4,4%</span> dari Baseline OpenAI.
              Kombinasi BM25 + Dense Retrieval (via RRF) + CrossEncoder Reranker.
            </p>
          </div>

          {/* Accuracy ranking */}
          <div>
            <h3 className="text-white font-semibold text-sm mb-3">
              Ranking berdasarkan Label Accuracy
            </h3>
            <div className="space-y-1.5">
              {sorted.map((cfg, idx) => {
                const stats = AGGREGATE_STATS[cfg.key]
                const deltaBl = stats.accuracy - AGGREGATE_STATS.baseline_openai.accuracy
                const deltaLl = stats.accuracy - AGGREGATE_STATS.baseline_llama.accuracy
                const pct = (stats.accuracy / 70) * 100
                const isSelected = cfg.key === selectedConfig
                return (
                  <div
                    key={cfg.key}
                    className={`flex items-center gap-3 px-3 py-2 rounded-md ${
                      isSelected
                        ? 'bg-violet-600/20 border border-violet-500/40'
                        : 'bg-gray-800/50 border border-transparent'
                    }`}
                  >
                    <span className="text-gray-500 text-xs font-mono w-6">
                      #{idx + 1}
                    </span>
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: cfg.color }}
                    />
                    <span className="text-xs text-gray-500 w-28 flex-shrink-0">
                      {cfg.model}
                    </span>
                    <span
                      className={`text-xs flex-1 ${
                        isSelected ? 'text-violet-200 font-medium' : 'text-gray-300'
                      }`}
                    >
                      {cfg.label}
                    </span>

                    {/* Bar */}
                    <div className="flex-1 max-w-xs h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${pct}%`,
                          backgroundColor: cfg.color,
                        }}
                      />
                    </div>

                    <span className="text-xs text-white font-mono font-bold w-14 text-right">
                      {stats.accuracy}%
                    </span>
                    <span className="text-[10px] text-gray-500 w-16 text-right font-mono">
                      {stats.correctCount}/500
                    </span>
                    <span
                      className={`text-[10px] font-mono w-16 text-right ${
                        deltaBl > 0 ? 'text-green-400' : deltaBl < 0 ? 'text-red-400' : 'text-gray-500'
                      }`}
                      title="vs Baseline OpenAI"
                    >
                      {deltaBl > 0 ? '+' : ''}
                      {deltaBl.toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Full metrics table */}
          <div>
            <h3 className="text-white font-semibold text-sm mb-3">
              Metrik Lengkap per Konfigurasi
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-400">
                    <th className="text-left px-3 py-2 font-semibold">Config</th>
                    <th className="text-right px-2 py-2 font-semibold">Accuracy</th>
                    <th className="text-right px-2 py-2 font-semibold">Faithful.</th>
                    <th className="text-right px-2 py-2 font-semibold">Ctx Recall</th>
                    <th className="text-right px-2 py-2 font-semibold">Ans Relev.</th>
                    <th className="text-right px-2 py-2 font-semibold">Ctx Prec.</th>
                  </tr>
                </thead>
                <tbody>
                  {CONFIGS.map((cfg) => {
                    const s = AGGREGATE_STATS[cfg.key]
                    const isSelected = cfg.key === selectedConfig
                    return (
                      <tr
                        key={cfg.key}
                        className={`border-b border-gray-800/50 ${
                          isSelected ? 'bg-violet-600/10' : ''
                        }`}
                      >
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-1.5 h-1.5 rounded-full"
                              style={{ backgroundColor: cfg.color }}
                            />
                            <span className="text-gray-300">{cfg.label}</span>
                            <span className="text-gray-500 text-[10px]">
                              ({cfg.model})
                            </span>
                          </div>
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-white font-bold">
                          {s.accuracy}%
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {s.faithfulness.toFixed(4)}
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {s.contextRecall.toFixed(4)}
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {s.answerRelevancy !== null
                            ? s.answerRelevancy.toFixed(4)
                            : '---'}
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {s.contextPrecision !== null
                            ? s.contextPrecision.toFixed(4)
                            : '---'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-label accuracy */}
          <div>
            <h3 className="text-white font-semibold text-sm mb-2">
              Akurasi Per-Label (Ground Truth: yes={GROUND_TRUTH_DIST.yes}, no=
              {GROUND_TRUTH_DIST.no}, maybe={GROUND_TRUTH_DIST.maybe})
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-400">
                    <th className="text-left px-3 py-2 font-semibold">Config</th>
                    <th className="text-right px-2 py-2 font-semibold text-green-400">yes (n=275)</th>
                    <th className="text-right px-2 py-2 font-semibold text-red-400">no (n=159)</th>
                    <th className="text-right px-2 py-2 font-semibold text-yellow-400">maybe (n=66)</th>
                  </tr>
                </thead>
                <tbody>
                  {CONFIGS.map((cfg) => {
                    const pl = PER_LABEL_ACCURACY[cfg.key]
                    return (
                      <tr key={cfg.key} className="border-b border-gray-800/50">
                        <td className="px-3 py-2 text-gray-300">
                          {cfg.label}{' '}
                          <span className="text-gray-500 text-[10px]">({cfg.model})</span>
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {pl.yes.correct}/{pl.yes.total} (
                          {((pl.yes.correct / pl.yes.total) * 100).toFixed(1)}%)
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {pl.no.correct}/{pl.no.total} (
                          {((pl.no.correct / pl.no.total) * 100).toFixed(1)}%)
                        </td>
                        <td className="text-right px-2 py-2 font-mono text-gray-300">
                          {pl.maybe.correct}/{pl.maybe.total} (
                          {((pl.maybe.correct / pl.maybe.total) * 100).toFixed(1)}%)
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Key insights */}
          <div className="p-4 bg-gray-800/50 border border-gray-700/50 rounded-lg">
            <h3 className="text-white font-semibold text-sm mb-2">Temuan Utama</h3>
            <ul className="text-xs text-gray-400 space-y-1.5 list-disc list-inside">
              <li>
                <span className="text-white font-medium">Hybrid Retrieval</span> memberikan
                peningkatan terbesar (+3,8% vs BL OpenAI), melampaui teknik QR dan CR
                masing-masing.
              </li>
              <li>
                <span className="text-white font-medium">Query Rewriting</span> tidak
                membantu pada model yang kuat (0%) dan merusak model kecil (-5,6% di
                Llama).
              </li>
              <li>
                <span className="text-white font-medium">Kapasitas model</span> adalah
                faktor dominan: BL OpenAI (65,4%) sudah jauh lebih baik dari konfigurasi
                terbaik Llama (55,6%).
              </li>
              <li>
                <span className="text-white font-medium">QR + CR</span> mencapai distribusi
                prediksi paling seimbang (akurasi "no" tertinggi = 66,7%), tapi menurunkan
                semua metrik RAGAS.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
