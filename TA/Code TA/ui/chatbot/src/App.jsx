import React, { useCallback, useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import PipelinePanel from './components/PipelinePanel'
import WelcomeState from './components/WelcomeState'
import ChatMessages from './components/ChatMessages'
import ChatInput from './components/ChatInput'
import { useConversations } from './hooks/useConversations'

/**
 * 3-column layout:
 *   ┌─────────┬───────────────────┬────────────┐
 *   │ Sidebar │  Header           │ Pipeline   │
 *   │ (lg+)   │  Welcome/Messages │ (xl+)      │
 *   │         │  ChatInput        │            │
 *   └─────────┴───────────────────┴────────────┘
 *
 * Mobile/tablet: both side panels collapse into drawers triggered from header.
 */
export default function App() {
  const {
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
    hasMessages,
  } = useConversations()

  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [pipelineOpen, setPipelineOpen] = useState(false)

  const handleSelectSample = useCallback(
    (question) => sendMessage(question),
    [sendMessage]
  )

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Left: conversations */}
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={selectConversation}
        onNewChat={newChat}
        onDelete={deleteConversation}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Center: main chat column */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
          onTogglePipeline={() => setPipelineOpen((v) => !v)}
          showPipelineToggle={Boolean(lastPipeline.pipeline)}
          activeTitle={activeConversation?.title}
        />

        {hasMessages ? (
          <ChatMessages messages={messages} isLoading={isLoading} error={error} />
        ) : (
          <WelcomeState onSelectSample={handleSelectSample} disabled={isLoading} />
        )}

        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>

      {/* Right: pipeline details */}
      <PipelinePanel
        pipeline={lastPipeline.pipeline}
        model={lastPipeline.model}
        isOpen={pipelineOpen}
        onClose={() => setPipelineOpen(false)}
      />
    </div>
  )
}
