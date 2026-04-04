// ===========================
// Centralized API Layer
// ===========================

const API_BASE = '/api/v1';

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch('/health');
    return res.ok;
  } catch {
    return false;
  }
}

export async function startSession(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/session/start`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    let msg = `HTTP ${res.status}`;
    try {
      const data = JSON.parse(text);
      msg = data.message || data.detail || msg;
    } catch {
      msg = text || msg;
    }
    throw new Error(msg);
  }

  return res.json();
}

export async function sendQuery(sessionId: string, query: string, signal?: AbortSignal) {
  const res = await fetch(`${API_BASE}/session/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, query }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    let msg = `HTTP ${res.status}`;
    try {
      const data = JSON.parse(text);
      msg = data.message || data.detail || msg;
    } catch {
      msg = text || msg;
    }
    throw new Error(msg);
  }

  return res.json();
}

export async function getSessionData(sessionId: string, page = 1, pageSize = 100) {
  const res = await fetch(
    `${API_BASE}/session/${sessionId}/data?page=${page}&page_size=${pageSize}`
  );
  if (!res.ok) throw new Error(`Failed to fetch data: HTTP ${res.status}`);
  return res.json();
}

export async function deleteSession(sessionId: string) {
  const res = await fetch(`${API_BASE}/session/${sessionId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Failed to delete session: HTTP ${res.status}`);
  return res.json();
}
