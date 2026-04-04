'use client';

import { useState } from 'react';
import { useSession } from '@/hooks/useSession';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import SuggestionsSidebar from './SuggestionsSidebar';

export default function ChatSection() {
  const { messages, isLoading, clearChat, clearSession } = useSession();
  const [showSuggestions, setShowSuggestions] = useState(false);

  return (
    <section className="chat-section" id="chatSection">
      <div className="section-header">
        <h2><i className="fas fa-comments" /> Data Analysis Chat</h2>
        <div className="chat-controls">
          <button className="chat-btn" onClick={clearChat} title="Clear Chat">
            <i className="fas fa-trash" />
          </button>
          <button className="chat-btn" title="Save Conversation">
            <i className="fas fa-save" />
          </button>
          <button className="chat-btn" onClick={() => setShowSuggestions(prev => !prev)} title="Toggle Suggestions">
            <i className="fas fa-magic" />
          </button>
          <button className="new-session-btn" onClick={clearSession}>
            <i className="fas fa-plus" /> New Session
          </button>
        </div>
      </div>
      <div className="chat-container">
        {showSuggestions && (
          <SuggestionsSidebar onClose={() => setShowSuggestions(false)} />
        )}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <MessageList messages={messages} isLoading={isLoading} />
          <ChatInput />
        </div>
      </div>
    </section>
  );
}
