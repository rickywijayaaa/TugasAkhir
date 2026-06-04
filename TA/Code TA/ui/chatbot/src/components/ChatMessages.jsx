import React, { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'

/**
 * Container untuk daftar pesan. Auto-scroll ke pesan paling baru.
 *
 * Props:
 *   messages  — array dari useChat()
 *   isLoading — true saat menunggu respons (tampilkan typing indicator)
 *   error     — error message opsional
 */
export default function ChatMessages({ messages, isLoading, error }) {
  const bottomRef = useRef(null)

  // Auto-scroll ke bawah saat ada pesan baru atau loading state berubah
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [messages, isLoading])

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            role={msg.role}
            content={msg.content}
            source={msg.source}
            confidence={msg.confidence}
          />
        ))}

        {/* Loading state — typing indicator sebagai pesan assistant placeholder */}
        {isLoading && (
          <MessageBubble role="assistant">
            <TypingIndicator />
          </MessageBubble>
        )}

        {/* Error state */}
        {error && (
          <div className="text-center text-sm text-red-600 bg-red-50 rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
