import React, { useState, useEffect, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import DocumentPanel from './components/DocumentPanel'
import DocumentModal from './components/DocumentModal'

function getConfigKey(useQR, useCR) {
  if (useQR && useCR) return 'qr_cr'
  if (useQR) return 'qr'
  if (useCR) return 'cr'
  return 'baseline'
}

function getPhaseDelays(useQR, useCR) {
  // Returns [phase1Delay, phase2Delay, phase3Delay, phase4Delay] cumulative from t=0
  if (!useQR && !useCR) return [80, 900, 980, 2400]
  if (useQR && !useCR)  return [1200, 2900, 2980, 4700]
  if (!useQR && useCR)  return [80, 900, 2500, 4200]
  return [800, 2500, 5000, 6800]
}

export default function App() {
  const [activeQuestion, setActiveQuestion] = useState(null)
  const [phase, setPhase] = useState(0)
  const [typedAnswer, setTypedAnswer] = useState('')
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [useQR, setUseQR] = useState(false)
  const [useCR, setUseCR] = useState(false)

  const timersRef = useRef([])
  const typewriterRef = useRef(null)

  const clearAllTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout)
    timersRef.current = []
    if (typewriterRef.current) {
      clearInterval(typewriterRef.current)
      typewriterRef.current = null
    }
  }, [])

  const runPipeline = useCallback(
    (question, qr, cr) => {
      clearAllTimers()

      setActiveQuestion(question)
      setPhase(0)
      setTypedAnswer('')

      const delays = getPhaseDelays(qr, cr)
      delays.forEach((delay, idx) => {
        const t = setTimeout(() => {
          setPhase(idx + 1)
        }, delay)
        timersRef.current.push(t)
      })
    },
    [clearAllTimers]
  )

  const handleSelectQuestion = useCallback(
    (question) => {
      if (activeQuestion?.id === question.id && phase > 0 && phase < 5) return
      runPipeline(question, useQR, useCR)
    },
    [activeQuestion, phase, useQR, useCR, runPipeline]
  )

  const handleToggleQR = useCallback(
    (newVal) => {
      setUseQR(newVal)
      if (activeQuestion) {
        runPipeline(activeQuestion, newVal, useCR)
      }
    },
    [activeQuestion, useCR, runPipeline]
  )

  const handleToggleCR = useCallback(
    (newVal) => {
      setUseCR(newVal)
      if (activeQuestion) {
        runPipeline(activeQuestion, useQR, newVal)
      }
    },
    [activeQuestion, useQR, runPipeline]
  )

  // Typewriter effect: runs when phase reaches 4
  useEffect(() => {
    if (phase !== 4 || !activeQuestion) return

    const fullText = activeQuestion.answer
    let idx = 0
    setTypedAnswer('')

    typewriterRef.current = setInterval(() => {
      idx++
      setTypedAnswer(fullText.slice(0, idx))

      if (idx >= fullText.length) {
        clearInterval(typewriterRef.current)
        typewriterRef.current = null
        const t = setTimeout(() => setPhase(5), 300)
        timersRef.current.push(t)
      }
    }, 12)

    return () => {
      if (typewriterRef.current) {
        clearInterval(typewriterRef.current)
        typewriterRef.current = null
      }
    }
  }, [phase, activeQuestion])

  // Cleanup on unmount
  useEffect(() => {
    return () => clearAllTimers()
  }, [clearAllTimers])

  const handleDocClick = useCallback((doc) => {
    setSelectedDoc(doc)
  }, [])

  const handleCloseModal = useCallback(() => {
    setSelectedDoc(null)
  }, [])

  const configKey = getConfigKey(useQR, useCR)

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      {/* Sidebar */}
      <Sidebar
        activeQuestion={activeQuestion}
        onSelectQuestion={handleSelectQuestion}
        useQR={useQR}
        useCR={useCR}
        onToggleQR={handleToggleQR}
        onToggleCR={handleToggleCR}
        configKey={configKey}
      />

      {/* Chat area */}
      <ChatArea
        activeQuestion={activeQuestion}
        phase={phase}
        typedAnswer={typedAnswer}
        onDocClick={handleDocClick}
        useQR={useQR}
        useCR={useCR}
        configKey={configKey}
      />

      {/* Document panel */}
      <DocumentPanel
        question={activeQuestion}
        phase={phase}
        onDocClick={handleDocClick}
        useCR={useCR}
        configKey={configKey}
      />

      {/* Document modal */}
      {selectedDoc && (
        <DocumentModal doc={selectedDoc} onClose={handleCloseModal} />
      )}
    </div>
  )
}
