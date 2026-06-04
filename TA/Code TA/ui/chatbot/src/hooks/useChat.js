import { useState, useCallback, useRef } from 'react'
import { askChat } from '../services/chatService'

/**
 * Custom hook untuk mengelola state percakapan chatbot.
 *
 * State:
 *   messages — array of { id, role: 'user'|'assistant', content, source?, timestamp }
 *   isLoading — true saat menunggu respons backend
 *   error    — string|null saat ada error
 *
 * Actions:
 *   sendMessage(text) — kirim pertanyaan & terima jawaban
 *   reset()           — clear semua percakapan, balik ke welcome state
 *   hasMessages       — true kalau ada minimal 1 pesan
 */
export function useChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Ref untuk auto-increment message id (lebih reliable daripada Date.now() di StrictMode)
  const idCounterRef = useRef(0)
  const nextId = () => {
    idCounterRef.current += 1
    return idCounterRef.current
  }

  const sendMessage = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed || isLoading) return

    setError(null)

    const userMessage = {
      id: nextId(),
      role: 'user',
      content: trimmed,
      timestamp: Date.now(),
    }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await askChat(trimmed)
      const assistantMessage = {
        id: nextId(),
        role: 'assistant',
        content: response.answer,
        source: response.source,
        timestamp: Date.now(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      console.error('Chat error:', err)
      setError('Something went wrong while processing your question. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [isLoading])

  const reset = useCallback(() => {
    setMessages([])
    setError(null)
    setIsLoading(false)
    idCounterRef.current = 0
  }, [])

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    reset,
    hasMessages: messages.length > 0,
  }
}
