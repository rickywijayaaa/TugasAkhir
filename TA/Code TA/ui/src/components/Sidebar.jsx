import React from 'react'
import { QUESTIONS } from '../data/mockData'

const labelColors = {
  yes: 'bg-green-400/15 text-green-400 border-green-400/30',
  maybe: 'bg-yellow-400/15 text-yellow-400 border-yellow-400/30',
  no: 'bg-red-400/15 text-red-400 border-red-400/30',
}

const configBadgeColors = {
  baseline: 'bg-gray-700/50 text-gray-400 border-gray-600/40',
  qr: 'bg-blue-600/20 text-blue-300 border-blue-500/30',
  cr: 'bg-teal-600/20 text-teal-300 border-teal-500/30',
  qr_cr: 'bg-violet-600/20 text-violet-300 border-violet-500/30',
}

const configLabelText = {
  baseline: 'Baseline',
  qr: '+Query Rewriting',
  cr: '+Context Reranking',
  qr_cr: '+QR + CR',
}

function Toggle({ label, enabled, onToggle }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-400 text-xs">{label}</span>
      <button
        onClick={() => onToggle(!enabled)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200 focus:outline-none ${
          enabled ? 'bg-violet-600' : 'bg-gray-700'
        }`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform duration-200 ${
            enabled ? 'translate-x-[18px]' : 'translate-x-[3px]'
          }`}
        />
      </button>
    </div>
  )
}

export default function Sidebar({ activeQuestion, onSelectQuestion, useQR, useCR, onToggleQR, onToggleCR, configKey }) {
  return (
    <div className="w-64 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col h-full">
      {/* Header */}
      <div className="px-5 pt-6 pb-4 border-b border-gray-800">
        <div className="flex items-center gap-2.5 mb-1">
          {/* Logo icon */}
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
            <p className="text-gray-500 text-xs leading-tight">PubMedQA Demo</p>
          </div>
        </div>
      </div>

      {/* Pipeline Config */}
      <div className="px-4 py-3 border-b border-gray-800">
        <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider mb-2.5">
          Pipeline Config
        </p>
        <div className="space-y-2.5">
          <Toggle label="Query Rewriting" enabled={useQR} onToggle={onToggleQR} />
          <Toggle label="Context Reranking" enabled={useCR} onToggle={onToggleCR} />
        </div>
        <div className="mt-3">
          <span
            className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded border ${configBadgeColors[configKey]}`}
          >
            {configLabelText[configKey]}
          </span>
        </div>
      </div>

      {/* Questions list */}
      <div className="flex-1 overflow-y-auto px-3 py-4">
        <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider px-2 mb-3">
          Sample Questions
        </p>
        <div className="space-y-1.5">
          {QUESTIONS.map((q) => {
            const isActive = activeQuestion?.id === q.id
            return (
              <button
                key={q.id}
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
                    {q.shortLabel}
                  </span>
                  <span
                    className={`flex-shrink-0 text-xs font-medium px-1.5 py-0.5 rounded border ${labelColors[q.groundTruth]}`}
                  >
                    {q.groundTruth}
                  </span>
                </div>
                <p className="text-gray-500 text-xs leading-tight line-clamp-2 group-hover:text-gray-400 transition-colors">
                  {q.query}
                </p>
              </button>
            )
          })}
        </div>
      </div>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <p className="text-gray-500 text-xs">Powered by BM25 + Ollama</p>
        </div>
      </div>
    </div>
  )
}
