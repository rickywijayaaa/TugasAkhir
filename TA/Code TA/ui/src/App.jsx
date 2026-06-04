import React, { useState, useEffect, useRef, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import DocumentPanel from './components/DocumentPanel'
import DocumentModal from './components/DocumentModal'
import StatsDashboard from './components/StatsDashboard'
import SHExplorer from './components/SHExplorer'
import { CONFIGS } from './data/realData'

function getPhaseDelays(config) {
  // Returns cumulative delays [phase1, phase2, phase3, phase4]
  // based on config complexity (QR adds, Hybrid adds, CR adds)
  const hasQR = config.useQR
  const hasHybrid = config.useHybrid
  const hasCR = config.useCrossEncoder

  let t = 80
  const delays = []
  if (hasQR) t += 900
  delays.push(t) // phase 1: query processing done
  t += hasHybrid ? 1400 : 800
  delays.push(t) // phase 2: retrieval done
  if (hasCR) t += 1200
  delays.push(t) // phase 3: rerank done
  t += 1700
  delays.push(t) // phase 4: generation
  return delays
}

export default function App() {
  const [activeQuestion, setActiveQuestion] = useState(null)
  const [phase, setPhase] = useState(0)
  const [typedAnswer, setTypedAnswer] = useState('')
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [selectedConfig, setSelectedConfig] = useState('hybrid_cr_openai')
  const [statsOpen, setStatsOpen] = useState(false)
  const [shExplorerOpen, setShExplorerOpen] = useState(false)

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
    (question, configKey) => {
      clearAllTimers()
      const config = CONFIGS.find((c) => c.key === configKey) || CONFIGS[3]

      setActiveQuestion(question)
      setPhase(0)
      setTypedAnswer('')

      const delays = getPhaseDelays(config)
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
      if (activeQuestion?.idx === question.idx && phase > 0 && phase < 5) return
      runPipeline(question, selectedConfig)
    },
    [activeQuestion, phase, selectedConfig, runPipeline]
  )

  const handleSelectConfig = useCallback(
    (newConfig) => {
      setSelectedConfig(newConfig)
      if (activeQuestion) {
        runPipeline(activeQuestion, newConfig)
      }
    },
    [activeQuestion, runPipeline]
  )

  // Typewriter effect
  useEffect(() => {
    if (phase !== 4 || !activeQuestion) return

    const cfg = activeQuestion.configs?.[selectedConfig]
    const fullText = cfg?.answer || 'Jawaban tidak tersedia.'
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
  }, [phase, activeQuestion, selectedConfig])

  useEffect(() => {
    return () => clearAllTimers()
  }, [clearAllTimers])

  const handleDocClick = useCallback((doc) => {
    setSelectedDoc(doc)
  }, [])

  const handleCloseModal = useCallback(() => {
    setSelectedDoc(null)
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      <Sidebar
        activeQuestion={activeQuestion}
        onSelectQuestion={handleSelectQuestion}
        selectedConfig={selectedConfig}
        onSelectConfig={handleSelectConfig}
        onOpenStats={() => setStatsOpen(true)}
        onOpenSHExplorer={() => setShExplorerOpen(true)}
      />

      <ChatArea
        activeQuestion={activeQuestion}
        phase={phase}
        typedAnswer={typedAnswer}
        onDocClick={handleDocClick}
        selectedConfig={selectedConfig}
      />

      <DocumentPanel
        question={activeQuestion}
        phase={phase}
        onDocClick={handleDocClick}
        selectedConfig={selectedConfig}
      />

      {selectedDoc && <DocumentModal doc={selectedDoc} onClose={handleCloseModal} />}
      {statsOpen && (
        <StatsDashboard
          onClose={() => setStatsOpen(false)}
          selectedConfig={selectedConfig}
        />
      )}
      {shExplorerOpen && <SHExplorer onClose={() => setShExplorerOpen(false)} />}
    </div>
  )
}
