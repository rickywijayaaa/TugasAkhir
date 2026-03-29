import React, { useState, useEffect } from 'react'
import { AGGREGATE_STATS } from '../data/mockData'

// Animated score/stat bar
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

function ScoreBar({ score, color = 'blue', animated = true }) {
  const [width, setWidth] = useState(0)

  useEffect(() => {
    if (!animated) {
      setWidth(score * 100)
      return
    }
    const t = setTimeout(() => setWidth(score * 100), 150)
    return () => clearTimeout(t)
  }, [score, animated])

  const colorClass = color === 'violet' ? 'bg-violet-500' : 'bg-blue-500'

  return (
    <div className="flex items-center gap-1.5 flex-1 min-w-0">
      <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClass} rounded-full score-bar`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`text-xs font-mono flex-shrink-0 ${color === 'violet' ? 'text-violet-400' : 'text-blue-400'}`}>
        {score.toFixed(3)}
      </span>
    </div>
  )
}

export default function DocumentPanel({ question, phase, onDocClick, useCR, configKey }) {
  const [activeTab, setActiveTab] = useState('documents')

  // Reset to documents tab when question changes
  useEffect(() => {
    setActiveTab('documents')
  }, [question?.id])

  const hasRetrieval = phase >= 2
  const hasReranking = phase >= 3 && useCR

  // Decide which list to display and how
  let docList = []

  if (hasReranking && question) {
    const docMap = {}
    question.retrievedDocs.forEach((d) => { docMap[d.id] = d })
    docList = question.rerankedDocs.map((r, idx) => ({
      rank: idx + 1,
      doc: docMap[r.docId],
      score: r.ceScore,
      scoreLabel: 'CE',
      scoreColor: 'violet',
      prevRank: r.prevRank,
    }))
  } else if (hasRetrieval && question) {
    docList = question.retrievedDocs.map((doc, idx) => ({
      rank: idx + 1,
      doc,
      score: doc.bm25Score,
      scoreLabel: 'BM25',
      scoreColor: 'blue',
      prevRank: null,
    }))
  }

  // Evaluation tab data
  const configNames = {
    baseline: 'Baseline',
    qr: '+Query Rewriting',
    cr: '+Context Reranking',
    qr_cr: '+QR + CR',
  }
  const configColors = {
    baseline: 'bg-gray-700/50 text-gray-300 border-gray-600/40',
    qr: 'bg-blue-600/20 text-blue-300 border-blue-500/30',
    cr: 'bg-teal-600/20 text-teal-300 border-teal-500/30',
    qr_cr: 'bg-violet-600/20 text-violet-300 border-violet-500/30',
  }
  const labelBadgeColors = {
    yes: 'bg-green-400/15 text-green-400 border-green-400/30',
    no: 'bg-red-400/15 text-red-400 border-red-400/30',
    maybe: 'bg-yellow-400/15 text-yellow-400 border-yellow-400/30',
  }

  return (
    <div className="w-80 flex-shrink-0 bg-gray-900 border-l border-gray-800 flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h2 className="text-white font-semibold text-sm">Documents & Evaluation</h2>
          <p className="text-gray-500 text-xs mt-0.5">
            {hasReranking
              ? 'Reranked order'
              : hasRetrieval
              ? 'BM25 order'
              : 'Awaiting retrieval'}
          </p>
        </div>
        {hasRetrieval && activeTab === 'documents' && (
          <span className="text-xs bg-gray-800 text-gray-400 border border-gray-700 px-2 py-0.5 rounded-full">
            {docList.length} docs
          </span>
        )}
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-gray-800 flex-shrink-0">
        <button
          onClick={() => setActiveTab('documents')}
          className={`flex-1 py-2.5 text-xs font-medium transition-colors ${
            activeTab === 'documents'
              ? 'text-white border-b-2 border-violet-500'
              : 'text-gray-500 hover:text-gray-400'
          }`}
        >
          Documents
        </button>
        <button
          onClick={() => setActiveTab('evaluation')}
          className={`flex-1 py-2.5 text-xs font-medium transition-colors relative ${
            activeTab === 'evaluation'
              ? 'text-white border-b-2 border-violet-500'
              : 'text-gray-500 hover:text-gray-400'
          }`}
        >
          Evaluation
          {phase === 5 && activeTab !== 'evaluation' && (
            <span className="absolute top-2 right-4 w-1.5 h-1.5 rounded-full bg-green-400" />
          )}
        </button>
      </div>

      {/* Tab content */}
      {activeTab === 'documents' ? (
        /* Documents tab */
        <div className="flex-1 overflow-y-auto">
          {/* Score type indicator */}
          {hasRetrieval && (
            <div className="px-4 py-2 border-b border-gray-800/50">
              <div className="flex items-center gap-1.5">
                {!hasReranking ? (
                  <>
                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                    <span className="text-gray-500 text-xs">Sorted by BM25 score</span>
                  </>
                ) : (
                  <>
                    <div className="w-2 h-2 rounded-full bg-violet-500" />
                    <span className="text-gray-500 text-xs">Sorted by cross-encoder score</span>
                  </>
                )}
              </div>
            </div>
          )}

          {!hasRetrieval ? (
            <div className="flex flex-col items-center justify-center h-full px-6 text-center">
              <div className="w-12 h-12 rounded-xl bg-gray-800 flex items-center justify-center mb-3">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-6 h-6 text-gray-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <p className="text-gray-500 text-sm font-medium">No documents yet</p>
              <p className="text-gray-600 text-xs mt-1 leading-relaxed">
                Documents will appear here after retrieval begins
              </p>
            </div>
          ) : (
            <div className="p-3 space-y-2">
              {docList.map((item) => {
                const moved = item.prevRank !== null ? item.rank - item.prevRank : 0
                const movedUp = moved < 0
                const movedDown = moved > 0
                const unchanged = moved === 0

                return (
                  <button
                    key={item.doc.id}
                    onClick={() => onDocClick({ ...item.doc, ceScore: item.scoreLabel === 'CE' ? item.score : undefined })}
                    className="w-full text-left bg-gray-800/40 hover:bg-gray-800 border border-gray-700/40 hover:border-gray-600/70 rounded-lg px-3 py-3 transition-all duration-150 group fade-in"
                  >
                    <div className="flex items-start gap-2.5">
                      {/* Rank bubble */}
                      <div className="flex flex-col items-center gap-0.5 flex-shrink-0 pt-0.5">
                        <span
                          className={`w-6 h-6 rounded-full text-xs flex items-center justify-center font-bold ${
                            item.scoreLabel === 'CE'
                              ? 'bg-violet-600/25 text-violet-300'
                              : 'bg-blue-600/25 text-blue-300'
                          }`}
                        >
                          {item.rank}
                        </span>
                        {/* Movement indicator (only after reranking) */}
                        {hasReranking && (
                          <span
                            className={`text-xs font-medium ${
                              movedUp
                                ? 'text-green-400'
                                : movedDown
                                ? 'text-red-400'
                                : 'text-gray-600'
                            }`}
                          >
                            {movedUp ? (
                              <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                              </svg>
                            ) : movedDown ? (
                              <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                            ) : (
                              '—'
                            )}
                          </span>
                        )}
                      </div>

                      {/* Doc info */}
                      <div className="flex-1 min-w-0">
                        <p className="text-gray-200 text-xs font-medium leading-snug group-hover:text-white transition-colors line-clamp-2">
                          {item.doc.title}
                        </p>
                        <p className="text-gray-500 text-xs mt-0.5 truncate">{item.doc.journal}</p>

                        {/* Score bar */}
                        <div className="mt-2 flex items-center gap-1.5">
                          <span className="text-gray-600 text-xs flex-shrink-0">{item.scoreLabel}</span>
                          <ScoreBar score={item.score} color={item.scoreColor} />
                        </div>

                        {/* Prev rank chip (after reranking) */}
                        {hasReranking && item.prevRank !== null && !unchanged && (
                          <div className="mt-1.5 flex items-center gap-1">
                            <span className="text-gray-600 text-xs">was rank</span>
                            <span className="text-gray-500 text-xs font-mono bg-gray-700/50 px-1.5 rounded">
                              #{item.prevRank}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      ) : (
        /* Evaluation tab */
        <div className="flex-1 overflow-y-auto">
          {phase < 5 || !question ? (
            <div className="flex flex-col items-center justify-center h-full px-6 text-center">
              <div className="w-10 h-10 rounded-xl bg-gray-800 flex items-center justify-center mb-3">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-5 h-5 text-gray-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <p className="text-gray-500 text-sm font-medium">No results yet</p>
              <p className="text-gray-600 text-xs mt-1">Select a question and run the pipeline to see evaluation metrics.</p>
            </div>
          ) : (() => {
            const configData = question.configs[configKey]
            const aggStats = AGGREGATE_STATS[configKey]
            const predicted = configData.label
            const gt = question.groundTruth
            const isCorrect = predicted === gt

            return (
              <div className="p-4 space-y-4">
                {/* Config badge */}
                <div>
                  <span
                    className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded border ${configColors[configKey]}`}
                  >
                    {configNames[configKey]}
                  </span>
                </div>

                {/* Prediction vs Ground Truth */}
                <div className="bg-gray-800/40 border border-gray-700/40 rounded-lg p-3 space-y-2">
                  <p className="text-gray-500 text-xs font-semibold uppercase tracking-wider">Prediction vs Ground Truth</p>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded border ${labelBadgeColors[predicted]}`}
                    >
                      {predicted.toUpperCase()}
                    </span>
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3 text-gray-600" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded border ${labelBadgeColors[gt]}`}
                    >
                      {gt.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {isCorrect ? (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        <span className="text-green-400 text-xs font-medium">Correct</span>
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                        <span className="text-red-400 text-xs font-medium">Incorrect</span>
                      </>
                    )}
                  </div>
                </div>

                {/* Aggregate Stats */}
                <div className="space-y-2">
                  <p className="text-gray-600 text-xs font-semibold uppercase tracking-wider">Dataset Stats · 500 Samples</p>
                  <div className="bg-gray-800/40 border border-gray-700/40 rounded-lg p-3 space-y-3">
                    {/* Label Accuracy */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Label Accuracy</span>
                        <span className="text-green-400 text-xs font-mono">{aggStats.accuracy.toFixed(1)}%</span>
                      </div>
                      <AnimatedBar value={aggStats.accuracy} maxValue={100} colorClass="bg-green-500" />
                    </div>
                    {/* Hallucination Rate */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Hallucination Rate</span>
                        <span className="text-red-400 text-xs font-mono">{aggStats.hallucinationRate.toFixed(1)}%</span>
                      </div>
                      <AnimatedBar value={aggStats.hallucinationRate} maxValue={100} colorClass="bg-red-500" />
                    </div>
                  </div>
                </div>

                {/* RAGAS Scores */}
                <div className="space-y-2">
                  <p className="text-gray-600 text-xs font-semibold uppercase tracking-wider">RAGAS Scores · This Question</p>
                  <div className="bg-gray-800/40 border border-gray-700/40 rounded-lg p-3 space-y-3">
                    {/* Faithfulness */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Faithfulness</span>
                        <span className="text-violet-400 text-xs font-mono">{configData.faithfulness.toFixed(3)}</span>
                      </div>
                      <AnimatedBar value={configData.faithfulness} maxValue={1} colorClass="bg-violet-500" />
                    </div>
                    {/* Answer Relevancy */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Answer Relevancy</span>
                        <span className="text-blue-400 text-xs font-mono">{configData.answerRelevancy.toFixed(3)}</span>
                      </div>
                      <AnimatedBar value={configData.answerRelevancy} maxValue={1} colorClass="bg-blue-500" />
                    </div>
                    {/* Context Precision */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Context Precision</span>
                        <span className="text-teal-400 text-xs font-mono">{configData.contextPrecision.toFixed(3)}</span>
                      </div>
                      <AnimatedBar value={configData.contextPrecision} maxValue={1} colorClass="bg-teal-500" />
                    </div>
                    {/* Context Recall */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400 text-xs">Context Recall</span>
                        <span className="text-indigo-400 text-xs font-mono">{configData.contextRecall.toFixed(3)}</span>
                      </div>
                      <AnimatedBar value={configData.contextRecall} maxValue={1} colorClass="bg-indigo-500" />
                    </div>
                  </div>
                </div>

                {/* Note */}
                <p className="text-gray-600 text-xs">
                  * Aggregate stats from 500-sample evaluation run.
                </p>
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}
