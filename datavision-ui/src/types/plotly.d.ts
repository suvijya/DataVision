declare module 'react-plotly.js' {
  import { Component } from 'react';

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    frames?: Plotly.Frame[];
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: Plotly.Figure, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: Plotly.Figure, graphDiv: HTMLElement) => void;
    onPurge?: (figure: Plotly.Figure, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
    onClick?: (event: Plotly.PlotMouseEvent) => void;
    onHover?: (event: Plotly.PlotMouseEvent) => void;
    onUnhover?: (event: Plotly.PlotMouseEvent) => void;
    onSelected?: (event: Plotly.PlotSelectionEvent) => void;
    onRelayout?: (event: Plotly.PlotRelayoutEvent) => void;
    revision?: number;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}

declare namespace Plotly {
  interface Data {
    type?: string;
    x?: unknown;
    y?: unknown;
    z?: unknown;
    values?: unknown;
    labels?: unknown;
    name?: string;
    mode?: string;
    [key: string]: unknown;
  }

  interface Layout {
    title?: string | { text: string };
    height?: number;
    width?: number;
    autosize?: boolean;
    paper_bgcolor?: string;
    plot_bgcolor?: string;
    font?: { color?: string; family?: string; size?: number };
    geo?: Record<string, unknown>;
    [key: string]: unknown;
  }

  interface Config {
    responsive?: boolean;
    displayModeBar?: boolean;
    displaylogo?: boolean;
    [key: string]: unknown;
  }

  interface Frame {
    [key: string]: unknown;
  }

  interface Figure {
    data: Data[];
    layout: Partial<Layout>;
    frames?: Frame[];
  }

  interface PlotMouseEvent {
    points: Array<{
      curveNumber: number;
      pointNumber: number;
      x: unknown;
      y: unknown;
      [key: string]: unknown;
    }>;
    event: MouseEvent;
  }

  interface PlotSelectionEvent {
    points: Array<{
      curveNumber: number;
      pointNumber: number;
      [key: string]: unknown;
    }>;
  }

  interface PlotRelayoutEvent {
    [key: string]: unknown;
  }
}
