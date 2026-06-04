import React, { useState, useRef, useEffect, useCallback } from 'react'

const MAX_LENGTH = 800

/**
 * Input bar bawah halaman: textarea auto-resize + tombol kirim.
 *
 * Props:
 *   onSend     — (text) => void
 *   disabled   — true saat loading; tombol & textarea di-disable
 *   placeholder — string opsional
 */
export default function ChatInput({ onSend, disabled, placeholder = 'Type your health question...' }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  // Auto-resize textarea sesuai konten (max ~6 baris)
  const adjustHeight = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    const maxHeight = 168 // ~6 baris @28px line-height
    el.style.height = Math.min(el.scrollHeight, maxHeight) + 'px'
  }, [])

  useEffect(() => {
    adjustHeight()
  }, [text, adjustHeight])

  const submit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
  }, [text, disabled, onSend])

  // Enter untuk kirim, Shift+Enter untuk newline
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white px-4 sm:px-6 py-4">
      <div className="max-w-3xl mx-auto">
        <div
          className={`flex items-end gap-2 rounded-2xl border bg-white px-3 py-2
                      transition-colors
                      ${
                        disabled
                          ? 'border-slate-200 bg-slate-50'
                          : 'border-slate-300 focus-within:border-slate-400 focus-within:shadow-sm'
                      }`}
        >
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_LENGTH))}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={placeholder}
            rows={1}
            className="flex-1 resize-none bg-transparent text-[15px] text-slate-900 placeholder-slate-400
                       py-2 px-1 outline-none disabled:cursor-not-allowed"
            style={{ minHeight: '28px', maxHeight: '168px' }}
            aria-label="Type your question"
          />

          <button
            type="button"
            onClick={submit}
            disabled={disabled || !text.trim()}
            className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center
                        transition-all duration-150
                        ${
                          disabled || !text.trim()
                            ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                            : 'bg-slate-900 text-white hover:bg-slate-800 active:scale-95 cursor-pointer'
                        }`}
            aria-label="Send question"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path
                d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"
                transform="rotate(90 10 10)"
              />
            </svg>
          </button>
        </div>

        {/* Footer hint */}
        <p className="text-center text-[11px] text-slate-400 mt-2">
          {disabled
            ? 'Processing your question...'
            : 'Press Enter to send · Shift+Enter for a new line'}
        </p>
      </div>
    </div>
  )
}
