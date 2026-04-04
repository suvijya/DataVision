'use client';

export default function LoadingOverlay({ text = 'Processing...', show }: { text?: string; show: boolean }) {
  if (!show) return null;

  return (
    <div className="loading-overlay show">
      <div className="loading-content">
        <div className="spinner" />
        <p>{text}</p>
      </div>
    </div>
  );
}
