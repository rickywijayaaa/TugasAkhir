import React from 'react'

/**
 * Kartu pertanyaan sampel — klik untuk auto-submit ke chat.
 *
 * Props:
 *   topic    — label kategori (mis. "Nutrisi", "Olahraga")
 *   icon     — emoji icon
 *   question — teks pertanyaan
 *   onClick  — handler saat kartu diklik
 *   disabled — saat sedang loading, kartu tidak bisa diklik
 */
export default function SampleCard({ topic, icon, question, onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`group text-left w-full p-4 rounded-xl border border-slate-200 bg-white
                  transition-all duration-200
                  ${
                    disabled
                      ? 'opacity-50 cursor-not-allowed'
                      : 'hover:border-slate-300 hover:shadow-sm hover:-translate-y-0.5 cursor-pointer'
                  }`}
    >
      <div className="flex items-start gap-3">
        <div
          className="flex-shrink-0 w-9 h-9 rounded-lg bg-slate-50 flex items-center justify-center
                     text-lg group-hover:bg-slate-100 transition-colors"
        >
          <span aria-hidden="true">{icon}</span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-1">
            {topic}
          </div>
          <p className="text-[13.5px] text-slate-800 leading-snug">{question}</p>
        </div>
        <div className="flex-shrink-0 text-slate-400 group-hover:text-slate-600 transition-colors mt-1">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      </div>
    </button>
  )
}
