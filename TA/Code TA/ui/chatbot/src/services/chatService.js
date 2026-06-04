/**
 * Chat service — calls the FastAPI backend at VITE_API_BASE_URL.
 *
 * On any network/HTTP error we fall back to the local mock (fuzzy match against
 * pre-crafted sample answers). This keeps the demo usable offline.
 *
 * Response shape (mirrors backend models.ChatResponse):
 *   { answer, pipeline: { original_query, rewritten_query, retrieved_docs, latency_ms }, model }
 * Mock fallback adds an extra `source` field (legacy) and no pipeline.
 */

import { SAMPLE_QUESTIONS } from '../data/sampleQuestions'

const DEFAULT_API_BASE =
  (import.meta.env && import.meta.env.VITE_API_BASE_URL) || 'http://127.0.0.1:8000'

const CHAT_ENDPOINT = `${DEFAULT_API_BASE.replace(/\/$/, '')}/api/chat`

// ---------- Real backend call ------------------------------------------------

async function callBackend(question) {
  // 60s timeout — pipeline normally finishes in ~3s, but cold start of the
  // cross-encoder model can take longer the first time.
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 60_000)

  try {
    const res = await fetch(CHAT_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    })

    if (!res.ok) {
      const errBody = await res.text().catch(() => '')
      throw new Error(`Backend returned ${res.status}: ${errBody || res.statusText}`)
    }

    const data = await res.json()
    return {
      answer: data.answer,
      pipeline: data.pipeline ?? null,
      confidence: data.confidence ?? null,
      model: data.model ?? null,
      source: null, // backend has rich pipeline.retrieved_docs instead
    }
  } finally {
    clearTimeout(timeout)
  }
}

// ---------- Mock fallback (kept for offline/demo) ----------------------------

function normalize(str) {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function similarity(queryStr, candidateStr) {
  const q = new Set(normalize(queryStr).split(' ').filter((w) => w.length >= 3))
  const c = new Set(normalize(candidateStr).split(' ').filter((w) => w.length >= 3))
  if (q.size === 0 || c.size === 0) return 0
  let overlap = 0
  q.forEach((w) => {
    if (c.has(w)) overlap += 1
  })
  return overlap / Math.max(q.size, c.size)
}

const SIMILARITY_THRESHOLD = 0.3

function findBestMatch(question) {
  let best = { sample: null, score: 0 }
  SAMPLE_QUESTIONS.forEach((sample) => {
    const score = similarity(question, sample.question)
    if (score > best.score) best = { sample, score }
  })
  return best.score >= SIMILARITY_THRESHOLD ? best : null
}

const FALLBACK_ANSWER = `Sorry, I couldn't find specific research on that question in my knowledge base. For topics I'm less familiar with, **it's wiser to consult a healthcare professional** who can give advice tailored to your personal condition.

Try picking one of the sample questions, or ask something about everyday health topics such as **nutrition, exercise, sleep, mental health, or basic medication** — I'll do my best to help.`

const BACKEND_DOWN_NOTICE = `_(Backend unavailable — answering from offline sample set.)_\n\n`

async function mockAnswer(question, { reason }) {
  // Simulate a little latency so the UX matches a real call.
  await new Promise((resolve) => setTimeout(resolve, 800 + Math.random() * 700))

  const match = findBestMatch(question)
  const prefix = reason ? BACKEND_DOWN_NOTICE : ''

  if (match) {
    return {
      answer: prefix + match.sample.answer,
      pipeline: null,
      confidence: null,
      model: 'mock',
      source: { pubid: match.sample.pubid, topic: match.sample.topic },
    }
  }
  return {
    answer: prefix + FALLBACK_ANSWER,
    pipeline: null,
    confidence: null,
    model: 'mock',
    source: null,
  }
}

// ---------- Public API -------------------------------------------------------

/**
 * Send a question to the backend (with mock fallback).
 * @param {string} question
 */
export async function askChat(question) {
  if (!question || !question.trim()) {
    return {
      answer: 'Please type a question first.',
      pipeline: null,
      confidence: null,
      model: null,
      source: null,
    }
  }

  try {
    return await callBackend(question.trim())
  } catch (err) {
    console.warn('[chatService] backend call failed, using mock:', err.message)
    return mockAnswer(question.trim(), { reason: err.message })
  }
}
