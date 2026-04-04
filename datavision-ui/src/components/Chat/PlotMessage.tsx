'use client';

import dynamic from 'next/dynamic';
import type { PlotData } from '@/types';
import { processTraces, processLayout } from '@/lib/plotly-utils';

// Dynamic import for Plotly (no SSR — it requires window)
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PlotMessageProps {
  plotData: PlotData;
}

export default function PlotMessage({ plotData }: PlotMessageProps) {
  const rawData = plotData.data;
  if (!rawData) return <p className="analysis-text">No chart data available.</p>;

  const rawTraces = rawData.data || [];
  const rawLayout = rawData.layout || {};

  const traces = processTraces(rawTraces);
  const layout = processLayout({
    ...rawLayout,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(30,30,30,0.8)',
    font: { color: '#e5e5e5', family: 'Inter, sans-serif' },
  } as Record<string, unknown>);

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    ...rawData.config,
  };

  return (
    <div className="plot-container">
      {plotData.title && <h4 className="analysis-header">{plotData.title}</h4>}
      <Plot
        data={traces as Plotly.Data[]}
        layout={layout as Partial<Plotly.Layout>}
        config={config as Partial<Plotly.Config>}
        style={{ width: '100%', minHeight: '400px' }}
        useResizeHandler
      />
      {plotData.insights && (
        <p className="analysis-text" style={{ marginTop: '0.5rem', fontStyle: 'italic' }}>
          {plotData.insights}
        </p>
      )}
    </div>
  );
}
