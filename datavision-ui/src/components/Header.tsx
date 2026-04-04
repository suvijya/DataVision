'use client';

import { useSession } from '@/hooks/useSession';

export default function Header() {
  const { sessionId, apiStatus } = useSession();

  const statusColor =
    apiStatus === 'ready' ? '#22c55e' : apiStatus === 'error' ? '#ef4444' : '#9b9b9b';
  const statusText =
    sessionId ? 'Session Active' : apiStatus === 'ready' ? 'Ready' : apiStatus === 'error' ? 'API Error' : 'Offline';

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <i className="fas fa-robot" />
          <h1>Data-Analysis Assistant</h1>
        </div>
        <div className="header-info">
          <span className="version">v2.0</span>
          <span className="status-indicator" style={{ color: statusColor }}>
            <i className="fas fa-circle" /> {statusText}
          </span>
        </div>
      </div>
    </header>
  );
}
