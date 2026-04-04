'use client';

import { useCallback, useState, useRef } from 'react';
import { useSession } from '@/hooks/useSession';

export default function FileUpload() {
  const { uploadFile } = useSession();
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);

      if (!file.name.toLowerCase().endsWith('.csv')) {
        setError('Invalid file type. Please select a CSV file.');
        return;
      }
      if (file.size > 16 * 1024 * 1024) {
        setError('File too large. Maximum size is 16MB.');
        return;
      }

      try {
        await uploadFile(file);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      }
    },
    [uploadFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  return (
    <section className="upload-section">
      <div
        className={`upload-area ${isDragging ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-icon">
          <i className="fas fa-cloud-upload-alt" />
        </div>
        <h2>Upload Your CSV Dataset</h2>
        <p>Drag & drop your CSV file here, or click to select</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          hidden
          onChange={(e) => {
            if (e.target.files?.[0]) handleFile(e.target.files[0]);
          }}
        />
        <button className="upload-btn" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
          <i className="fas fa-file-csv" /> Choose CSV File
        </button>
      </div>

      {error && (
        <div style={{ color: 'var(--danger)', textAlign: 'center', marginTop: '1rem' }}>
          <i className="fas fa-exclamation-circle" /> {error}
        </div>
      )}

      <div className="upload-info">
        <p><i className="fas fa-info-circle" /> Maximum file size: 16MB • Supported formats: CSV</p>
        <p><i className="fas fa-shield-alt" /> Your data is processed securely and never stored permanently</p>
      </div>
    </section>
  );
}
