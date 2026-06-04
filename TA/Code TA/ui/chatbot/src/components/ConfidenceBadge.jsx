import React from 'react'

/**
 * Color-coded badge that surfaces the retrieval confidence guard-rail result.
 *
 * Mapping:
 *   high     → green   (≥ 0.7)
 *   moderate → amber   (0.4 – 0.7)
 *   low      → red     (< 0.4, system refused to call the LLM)
 *
 * Props:
 *   confidence — { score, level, refused, breakdown, thresholds } from /api/chat
 *
 * If `confidence` is null (e.g. mock response) the badge renders nothing — the
 * intent is to make the guard rail visible only when it's a real backend call.
 */
export default function ConfidenceBadge({ confidence }) {
  if (!confidence) return null

  const { level, score, refused } = confidence
  const pct = Math.round((score ?? 0) * 100)
  const cfg = STYLES[level] || STYLES.low

  return (
    <div
      className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-medium border
                  ${cfg.bg} ${cfg.text} ${cfg.border}`}
      title={tooltipFor(confidence)}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} aria-hidden="true" />
      <span>{cfg.label}</span>
      <span className="text-slate-400">·</span>
      <span className="tabular-nums">{pct}%</span>
      {refused && (
        <>
          <span className="text-slate-400">·</span>
          <span className="uppercase tracking-wider text-[10px] font-semibold">
            LLM skipped
          </span>
        </>
      )}
    </div>
  )
}

const STYLES = {
  high: {
    label: 'High confidence',
    bg: 'bg-emerald-50',
    text: 'text-emerald-800',
    border: 'border-emerald-200',
    dot: 'bg-emerald-500',
  },
  moderate: {
    label: 'Moderate confidence',
    bg: 'bg-amber-50',
    text: 'text-amber-800',
    border: 'border-amber-200',
    dot: 'bg-amber-500',
  },
  low: {
    label: 'Low confidence',
    bg: 'bg-red-50',
    text: 'text-red-800',
    border: 'border-red-200',
    dot: 'bg-red-500',
  },
}

function tooltipFor(confidence) {
  const b = confidence.breakdown || {}
  const lines = [
    `Composite score: ${(confidence.score * 100).toFixed(1)}%`,
    `Level: ${confidence.level}`,
    confidence.refused ? 'LLM call skipped (guard rail engaged)' : 'LLM call made',
    '',
    'Signals:',
    `  rerank (cross-encoder): ${((b.rerank_signal ?? 0) * 100).toFixed(1)}%`,
    `  dense (similarity):     ${((b.dense_signal ?? 0) * 100).toFixed(1)}%`,
    `  consensus (top-3):      ${((b.consensus_signal ?? 0) * 100).toFixed(1)}%`,
  ]
  return lines.join('\n')
}
