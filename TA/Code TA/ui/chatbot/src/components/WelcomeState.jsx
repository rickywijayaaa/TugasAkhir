import React from 'react'
import SampleCard from './SampleCard'
import { SAMPLE_QUESTIONS } from '../data/sampleQuestions'

/**
 * Welcome state — ditampilkan saat belum ada percakapan.
 *
 * Berisi:
 *   - Heading + tagline
 *   - Grid 6 sample question cards (responsive: 1 col mobile, 2 col tablet/desktop)
 */
export default function WelcomeState({ onSelectSample, disabled }) {
  return (
    <div className="flex-1 flex items-center justify-center px-4 py-8 overflow-y-auto">
      <div className="w-full max-w-3xl mx-auto fade-in">
        {/* Heading */}
        <div className="text-center mb-10">
          <div
            className="inline-flex items-center justify-center w-12 h-12 rounded-2xl
                       bg-gradient-to-br from-slate-100 to-slate-200 mb-4"
          >
            <span aria-hidden="true" className="text-2xl">
              🧬
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900 mb-2">
            Biomedical Q&A
          </h1>
          <p className="text-slate-600 text-[15px] max-w-xl mx-auto leading-relaxed">
            A research-backed assistant that answers your health questions in a natural,
            evidence-based way — drawing on the PubMed database.
          </p>
        </div>

        {/* Sample questions grid */}
        <div className="mb-2">
          <p className="text-[13px] font-medium text-slate-500 uppercase tracking-wider mb-3">
            Try one of these questions
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {SAMPLE_QUESTIONS.map((sample) => (
              <SampleCard
                key={sample.pubid}
                topic={sample.topic}
                icon={sample.icon}
                question={sample.question}
                onClick={() => onSelectSample(sample.question)}
                disabled={disabled}
              />
            ))}
          </div>
        </div>

        {/* Footer hint */}
        <p className="text-center text-xs text-slate-400 mt-8 leading-relaxed">
          Sample questions are drawn from the{' '}
          <span className="font-medium text-slate-500">PubMedQA dataset</span>. You can also type
          your own question in the box below.
        </p>
      </div>
    </div>
  )
}
