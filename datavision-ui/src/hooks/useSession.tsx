'use client';

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import type { DataPreview, ChatMessage, QueryResponse, PlotData } from '@/types';
import * as api from '@/lib/api';

interface SessionContextValue {
  sessionId: string | null;
  dataPreview: DataPreview | null;
  messages: ChatMessage[];
  isLoading: boolean;
  loadingText: string;
  apiStatus: 'ready' | 'error' | 'offline';
  uploadFile: (file: File) => Promise<void>;
  sendQuery: (query: string) => Promise<void>;
  clearChat: () => void;
  clearSession: () => void;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataPreview, setDataPreview] = useState<DataPreview | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('Processing...');
  const [apiStatus, setApiStatus] = useState<'ready' | 'error' | 'offline'>('offline');
  const abortRef = useRef<AbortController | null>(null);

  // Check API on mount
  useEffect(() => {
    api.healthCheck().then((ok) => setApiStatus(ok ? 'ready' : 'error'));
  }, []);

  const uploadFile = useCallback(async (file: File) => {
    setIsLoading(true);
    setLoadingText('Uploading and processing your CSV file...');
    try {
      const result = await api.startSession(file);
      setSessionId(result.session_id);
      setDataPreview(result.data_preview);
      setMessages([
        {
          id: 'welcome',
          role: 'assistant',
          content: `👋 I've analyzed your dataset "${file.name}" (${result.data_preview.shape[0]} rows × ${result.data_preview.shape[1]} columns). Ask me anything about your data!`,
          type: 'text',
          timestamp: new Date(),
        },
      ]);
      setApiStatus('ready');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      throw new Error(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const sendQuery = useCallback(
    async (query: string) => {
      if (!sessionId) return;

      // Add user message
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: query,
        type: 'text',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);

      setIsLoading(true);
      setLoadingText('Analyzing your query...');

      try {
        abortRef.current = new AbortController();
        const timeoutId = setTimeout(() => abortRef.current?.abort(), 120000);

        const response: QueryResponse = await api.sendQuery(
          sessionId,
          query,
          abortRef.current.signal
        );
        clearTimeout(timeoutId);

        const assistantMsg = buildAssistantMessage(response);
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        let errorContent = '❌ An error occurred.';
        if (err instanceof Error) {
          if (err.name === 'AbortError') {
            errorContent = '⏱️ The analysis timed out. Try a simpler query.';
          } else {
            errorContent = `❌ ${err.message}`;
          }
        }
        setMessages((prev) => [
          ...prev,
          {
            id: `error-${Date.now()}`,
            role: 'assistant',
            content: errorContent,
            type: 'error',
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
  }, []);

  const clearSession = useCallback(() => {
    if (sessionId) {
      api.deleteSession(sessionId).catch(console.error);
    }
    setSessionId(null);
    setDataPreview(null);
    setMessages([]);
  }, [sessionId]);

  return (
    <SessionContext.Provider
      value={{
        sessionId,
        dataPreview,
        messages,
        isLoading,
        loadingText,
        apiStatus,
        uploadFile,
        sendQuery,
        clearChat,
        clearSession,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used within a SessionProvider');
  return ctx;
}

// ===========================
// Helpers
// ===========================

function buildAssistantMessage(response: QueryResponse): ChatMessage {
  const { response_type, data, message } = response;

  switch (response_type) {
    case 'plot': {
      let plotData: PlotData;
      const raw = data as Record<string, unknown>;

      if (raw['application/vnd.plotly.v1+json']) {
        plotData = {
          data: raw['application/vnd.plotly.v1+json'] as PlotData['data'],
          title: raw.title as string,
          type: raw.type as string,
          insights: (raw.insights as string) || message,
        };
      } else {
        plotData = {
          data: raw.data as PlotData['data'],
          title: raw.title as string,
          type: raw.type as string,
          insights: (raw.insights as string) || message,
        };
      }

      return {
        id: `plot-${Date.now()}`,
        role: 'assistant',
        content: plotData.insights || plotData.title || 'Chart generated',
        type: 'plot',
        plotData,
        timestamp: new Date(),
      };
    }

    case 'statistics':
    case 'insight':
    case 'text': {
      const raw = data as Record<string, unknown>;
      const text = (raw.interpretation as string) || (raw.text as string) || message;
      return {
        id: `text-${Date.now()}`,
        role: 'assistant',
        content: text,
        type: 'text',
        timestamp: new Date(),
      };
    }

    case 'error': {
      const raw = data as Record<string, unknown>;
      return {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `❌ ${(raw.message as string) || message}`,
        type: 'error',
        timestamp: new Date(),
      };
    }

    default:
      return {
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: message || 'Analysis completed',
        type: 'text',
        timestamp: new Date(),
      };
  }
}
