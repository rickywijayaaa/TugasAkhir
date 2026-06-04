import { useCallback, useEffect, useRef, useState } from 'react'
import { askChat } from '../services/chatService'

const STORAGE_KEY = 'biorag.conversations.v1'
const MAX_CONVERSATIONS = 50

/**
 * Manages a list of conversations persisted to localStorage.
 *
 * Each conversation:
 *   { id, title, createdAt, updatedAt, messages: [{id, role, content, source?, pipeline?, model?}] }
 *
 * Title is derived from the first user message (truncated to 60 chars).
 * On mount we hydrate from localStorage; every state change writes back.
 */
export function useConversations() {
  const [conversations, setConversations] = useState(() => loadFromStorage())
  const [activeId, setActiveId] = useState(() => {
    const all = loadFromStorage()
    return all[0]?.id ?? null
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Counter for unique message ids within a conversation.
  const msgIdRef = useRef(0)
  const nextMsgId = () => {
    msgIdRef.current += 1
    return msgIdRef.current
  }

  // Persist whenever conversations change.
  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
    } catch (e) {
      console.warn('localStorage write failed', e)
    }
  }, [conversations])

  const activeConversation = conversations.find((c) => c.id === activeId) || null
  const messages = activeConversation?.messages ?? []

  /** Most recent assistant message that has pipeline data attached. */
  const lastPipeline = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant' && messages[i].pipeline) {
        return { pipeline: messages[i].pipeline, model: messages[i].model }
      }
    }
    return { pipeline: null, model: null }
  })()

  /**
   * Send a user message. Creates a new conversation if there's no active one.
   * Streams the assistant response back.
   */
  const sendMessage = useCallback(
    async (text) => {
      const trimmed = text.trim()
      if (!trimmed || isLoading) return
      setError(null)

      // Ensure we have an active conversation, creating one if needed.
      let convId = activeId
      if (!convId) {
        convId = createConversationId()
        const newConv = {
          id: convId,
          title: deriveTitle(trimmed),
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
        }
        setConversations((prev) => [newConv, ...prev].slice(0, MAX_CONVERSATIONS))
        setActiveId(convId)
      }

      const userMsg = {
        id: nextMsgId(),
        role: 'user',
        content: trimmed,
        timestamp: Date.now(),
      }
      appendMessage(convId, userMsg, { setConversations, deriveTitleFrom: trimmed })

      setIsLoading(true)
      try {
        const response = await askChat(trimmed)
        const assistantMsg = {
          id: nextMsgId(),
          role: 'assistant',
          content: response.answer,
          source: response.source ?? null,
          pipeline: response.pipeline ?? null,
          confidence: response.confidence ?? null,
          model: response.model ?? null,
          timestamp: Date.now(),
        }
        appendMessage(convId, assistantMsg, { setConversations })
      } catch (err) {
        console.error('chat error', err)
        setError('Something went wrong while processing your question. Please try again.')
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, isLoading]
  )

  const newChat = useCallback(() => {
    setActiveId(null)
    setError(null)
  }, [])

  const selectConversation = useCallback((id) => {
    setActiveId(id)
    setError(null)
  }, [])

  const deleteConversation = useCallback(
    (id) => {
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (activeId === id) setActiveId(null)
    },
    [activeId]
  )

  return {
    conversations,
    activeId,
    activeConversation,
    messages,
    lastPipeline,
    isLoading,
    error,
    sendMessage,
    newChat,
    selectConversation,
    deleteConversation,
    hasMessages: messages.length > 0,
  }
}

// ---------- helpers ---------------------------------------------------------

function loadFromStorage() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch (e) {
    console.warn('localStorage read failed', e)
    return []
  }
}

function createConversationId() {
  // Plain timestamp + 4 random alphanumeric chars — no crypto dep needed.
  const rand = Math.random().toString(36).slice(2, 6)
  return `c_${Date.now()}_${rand}`
}

function deriveTitle(firstMessage) {
  const cleaned = firstMessage.replace(/\s+/g, ' ').trim()
  return cleaned.length > 60 ? cleaned.slice(0, 60).trimEnd() + '…' : cleaned
}

function appendMessage(convId, message, { setConversations, deriveTitleFrom }) {
  setConversations((prev) =>
    prev.map((c) => {
      if (c.id !== convId) return c
      // Only set title once from the first user message.
      const title = deriveTitleFrom && !c.messages.some((m) => m.role === 'user')
        ? deriveTitle(deriveTitleFrom)
        : c.title
      return {
        ...c,
        title,
        updatedAt: Date.now(),
        messages: [...c.messages, message],
      }
    })
  )
}
