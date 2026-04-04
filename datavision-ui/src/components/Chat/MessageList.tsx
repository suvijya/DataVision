'use client';

import { useRef, useEffect } from 'react';
import type { ChatMessage } from '@/types';
import { formatAnalysisText } from '@/lib/formatters';
import PlotMessage from './PlotMessage';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

export default function MessageList({ messages, isLoading }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="messages">
      {messages.map((msg) => (
        <div key={msg.id} className={`message ${msg.role}`}>
          <div className="message-avatar">
            <i className={`fas ${msg.role === 'user' ? 'fa-user' : 'fa-robot'}`} />
          </div>
          <div className="message-content">
            {msg.type === 'plot' && msg.plotData ? (
              <>
                <PlotMessage plotData={msg.plotData} />
                {msg.content && (
                  <div
                    className="message-text"
                    dangerouslySetInnerHTML={{ __html: formatAnalysisText(msg.content) }}
                  />
                )}
              </>
            ) : (
              <div
                className="message-text"
                dangerouslySetInnerHTML={{ __html: formatAnalysisText(msg.content) }}
              />
            )}
            <div className="message-time">
              {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="message assistant">
          <div className="message-avatar">
            <i className="fas fa-robot" />
          </div>
          <div className="message-content">
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
