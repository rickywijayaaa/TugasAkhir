import React from 'react'

/**
 * Left sidebar: new-chat button + list of past conversations.
 *
 * Props:
 *   conversations       — [{id, title, updatedAt, messages}]
 *   activeId            — id of the currently open conversation, or null
 *   onSelect(id)        — switch to a conversation
 *   onNewChat()         — start a fresh conversation
 *   onDelete(id)        — remove a conversation
 *   isOpen              — controls drawer visibility on mobile
 *   onClose()           — close drawer (mobile)
 */
export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNewChat,
  onDelete,
  isOpen,
  onClose,
}) {
  return (
    <>
      {/* Backdrop for mobile drawer */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/30 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed lg:static top-0 left-0 z-40 h-full w-72 flex-shrink-0
                    bg-slate-50 border-r border-slate-200
                    flex flex-col transition-transform duration-200
                    ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
      >
        {/* Header inside sidebar */}
        <div className="flex-shrink-0 p-3 border-b border-slate-200">
          <button
            type="button"
            onClick={() => {
              onNewChat()
              onClose?.()
            }}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg
                       bg-white border border-slate-200 text-slate-700
                       hover:bg-slate-100 hover:border-slate-300 transition-colors
                       text-[14px] font-medium"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-4 h-4"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z"
                clipRule="evenodd"
              />
            </svg>
            New chat
          </button>
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto px-2 py-2">
          {conversations.length === 0 ? (
            <p className="text-center text-[12.5px] text-slate-400 py-8 px-4 leading-relaxed">
              Your conversations will appear here.
            </p>
          ) : (
            <ul className="space-y-1">
              {conversations.map((c) => (
                <li key={c.id}>
                  <ConversationItem
                    conversation={c}
                    isActive={c.id === activeId}
                    onClick={() => {
                      onSelect(c.id)
                      onClose?.()
                    }}
                    onDelete={() => onDelete(c.id)}
                  />
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Footer */}
        <div className="flex-shrink-0 p-3 border-t border-slate-200 text-[11px] text-slate-500">
          <p className="leading-relaxed">
            <span className="font-medium text-slate-700">BioRAG</span> · Thesis demo
          </p>
          <p className="mt-0.5 text-slate-400">Hybrid + QR + CR · GPT-4.1-mini</p>
        </div>
      </aside>
    </>
  )
}

/**
 * Single row in the conversation list.
 * Shows truncated title and a delete button on hover.
 */
function ConversationItem({ conversation, isActive, onClick, onDelete }) {
  return (
    <div
      className={`group relative rounded-lg transition-colors
                  ${isActive ? 'bg-slate-200/80' : 'hover:bg-slate-100'}`}
    >
      <button
        type="button"
        onClick={onClick}
        className="w-full text-left px-3 py-2 pr-8 rounded-lg"
        aria-label={`Open conversation: ${conversation.title}`}
      >
        <p className="text-[13px] text-slate-800 truncate leading-snug">
          {conversation.title || 'New conversation'}
        </p>
      </button>

      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          if (window.confirm('Delete this conversation?')) onDelete()
        }}
        className="absolute right-1.5 top-1/2 -translate-y-1/2 w-6 h-6 rounded
                   flex items-center justify-center
                   text-slate-400 hover:text-red-600 hover:bg-red-50
                   opacity-0 group-hover:opacity-100 focus:opacity-100
                   transition-opacity"
        aria-label="Delete conversation"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="w-3.5 h-3.5"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </div>
  )
}
