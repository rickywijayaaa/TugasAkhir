import React from 'react'

/**
 * Three-dot typing indicator dengan animasi pulse.
 * Dipakai sebagai pesan assistant placeholder saat menunggu respons.
 */
export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-2" aria-label="Assistant is typing">
      <span className="typing-dot w-2 h-2 rounded-full bg-slate-400" />
      <span className="typing-dot w-2 h-2 rounded-full bg-slate-400" />
      <span className="typing-dot w-2 h-2 rounded-full bg-slate-400" />
    </div>
  )
}
