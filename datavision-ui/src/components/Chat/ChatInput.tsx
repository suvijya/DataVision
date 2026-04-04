'use client';

import { useState, useRef, useEffect } from 'react';
import { useSession } from '@/hooks/useSession';

export default function ChatInput() {
  const { sendQuery, isLoading, sessionId } = useSession();
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = query.trim().length > 0 && !isLoading && !!sessionId;

  const handleSend = () => {
    if (!canSend) return;
    sendQuery(query.trim());
    setQuery('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    }
  }, [query]);

  return (
    <div className="chat-input">
      <div className="input-toolbar">
        <button className="toolbar-btn" title="Voice Input">
          <i className="fas fa-microphone" />
        </button>
        <button className="toolbar-btn" title="Attach Image">
          <i className="fas fa-camera" />
        </button>
        <button className="toolbar-btn" title="Query Templates">
          <i className="fas fa-code" />
        </button>
        <div className="toolbar-divider" />
        <button className="toolbar-btn" title="Query History">
          <i className="fas fa-history" />
        </button>
      </div>
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            id="queryInput"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a question about your data..."
            maxLength={1000}
            rows={1}
          />
          <div className="input-actions">
            {query && (
              <button className="action-btn" onClick={() => setQuery('')} title="Clear">
                <i className="fas fa-times" />
              </button>
            )}
            <button className="action-btn" onClick={() => navigator.clipboard.readText().then(t => setQuery(q => q + t)).catch(() => {})} title="Paste">
              <i className="fas fa-paste" />
            </button>
          </div>
        </div>
        <button id="sendBtn" disabled={!canSend} onClick={handleSend}>
          <i className="fas fa-paper-plane" />
        </button>
      </div>
      <div className="input-info">
        <div className="info-left">
          <span className="char-count" id="charCount">{query.length}/1000</span>
          <span className="query-type" id="queryType" />
        </div>
        <div className="info-right">
          <button className="info-btn">
            <i className="fas fa-question-circle" /> Examples
          </button>
          <button className="info-btn">
            <i className="fas fa-keyboard" /> Shortcuts
          </button>
        </div>
      </div>
    </div>
  );
}
