import React from 'react'
import { SAMPLES, CONFIGS, AGGREGATE_STATS } from '../data/realData'

const labelColors = {
  yes: 'bg-green-400/15 text-green-400 border-green-400/30',
  maybe: 'bg-yellow-400/15 text-yellow-400 border-yellow-400/30',
  no: 'bg-red-400/15 text-red-400 border-red-400/30',
}

export default function Sidebar({
  activeQuestion,
  onSelectQuestion,
  selectedConfig,
  onSelectConfig,
  onOpenStats,
}) {
  const currentConfig = CONFIGS.find((c) => c.key === selectedConfig) || CONFIGS[3]
  const currentStats = AGGREGATE_STATS[selectedConfig]

  // Group configs by model
  const llamaConfigs = CONFIGS.filter((c) => c.model === 'Llama 3.2')
  const openaiConfigs = CONFIGS.filter((c) => c.model === 'GPT-4.1-mini')

  return (
    <div className="w-72 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col h-full">
      {/* Header */}
      <div className="px-5 pt-6 pb-4 border-b border-gray-800">
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-7 h-7 rounded-lg bg-violet-600 flex items-center justify-center flex-shrink-0">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-4 h-4 text-white"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
            </svg>
          </div>
          <div>
            <h1 className="text-white font-semibold text-sm leading-tight">
              RAG Visualizer
            </h1>
            <p className="text-gray-500 text-xs leading-tight">
              PubMedQA · 9 Konfigurasi
            </p>
          </div>
        </div>
      </div>

      {/* Pipeline Config Selector */}
      <div className="px-4 py-3 border-b border-gray-800">
        <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2.5">
          Konfigurasi Pipeline
        </p>
        <select
          value={selectedConfig}
          onChange={(e) => onSelectConfig(e.target.value)}
          className="w-full bg-gray-800 text-gray-200 border border-gray-700 rounded-md px-2.5 py-1.5 text-xs focus:outline-none focus:border-violet-500"
        >
          <optgroup label="Llama 3.2 (3B)">
            {llamaConfigs.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </optgroup>
          <optgroup label="GPT-4.1-mini (OpenAI)">
            {openaiConfigs.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </optgroup>
        </select>

        {/* Current stats snippet */}
        {currentStats && (
          <div className="mt-3 p-2.5 rounded-md bg-gray-800/50 border border-gray-700/50">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">
                Model
              </span>
              <span className="text-xs text-gray-300 font-medium">
                {currentConfig.model}
              </span>
            </div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">
                Akurasi
              </span>
              <span
                className="text-xs font-bold"
                style={{ color: currentConfig.color }}
              >
                {currentStats.accuracy}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400 text-[10px] uppercase tracking-wider font-semibold">
                Faithfulness
              </span>
              <span className="text-xs text-gray-300 font-mono">
                {currentStats.faithfulness.toFixed(3)}
              </span>
            </div>
          </div>
        )}

        <button
          onClick={onOpenStats}
          className="mt-3 w-full flex items-center justify-center gap-1.5 bg-violet-600/10 hover:bg-violet-600/20 text-violet-300 text-xs font-medium py-1.5 rounded-md border border-violet-500/30 transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-3.5 h-3.5"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
          </svg>
          Lihat Perbandingan Penuh
        </button>
      </div>

      {/* Questions list */}
      <div className="flex-1 overflow-y-auto px-3 py-4">
        <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider px-2 mb-3">
          Contoh Pertanyaan (5 sampel real)
        </p>
        <div className="space-y-1.5">
          {SAMPLES.map((q) => {
            const isActive = activeQuestion?.idx === q.idx
            const cfgResult = q.configs[selectedConfig]
            const label = cfgResult?.label || q.ground_truth
            const isCorrect = cfgResult?.is_correct
            return (
              <button
                key={q.idx}
                onClick={() => onSelectQuestion(q)}
                className={`w-full text-left px-3 py-2.5 rounded-lg transition-all duration-150 group ${
                  isActive
                    ? 'bg-violet-600/20 border border-violet-500/40'
                    : 'hover:bg-gray-800 border border-transparent'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-0.5">
                  <span
                    className={`text-xs font-semibold truncate ${
                      isActive ? 'text-violet-300' : 'text-gray-300'
                    }`}
                  >
                    {q.short_label}
                  </span>
                  <div className="flex items-center gap-1 flex-shrink-0">
                    <span
                      className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${labelColors[q.ground_truth]}`}
                      title="Ground truth"
                    >
                      GT: {q.ground_truth}
                    </span>
                    {cfgResult?.label && (
                      <span
                        className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${
                          isCorrect
                            ? 'bg-green-400/10 text-green-400 border-green-400/20'
                            : 'bg-red-400/10 text-red-400 border-red-400/20'
                        }`}
                        title={isCorrect ? 'Benar' : 'Salah'}
                      >
                        {isCorrect ? '✓' : '✗'}
                      </span>
                    )}
                  </div>
                </div>
                <p className="text-gray-500 text-xs leading-tight line-clamp-2 group-hover:text-gray-400 transition-colors">
                  {q.question}
                </p>
              </button>
            )
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-800">
        <div className="flex items-center gap-1.5 mb-1">
          <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <p className="text-gray-500 text-[11px]">
            500 sampel · 9 konfigurasi
          </p>
        </div>
        <p className="text-gray-600 text-[10px]">
          Terbaik: Hybrid+CR 69,8%
        </p>
      </div>
    </div>
  )
}
