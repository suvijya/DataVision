// ===========================
// TypeScript Types for DataVision
// ===========================

export interface ColumnInfo {
  name: string;
  dtype: string;
  non_null_count: number;
  null_count: number;
  null_percentage: number;
  unique_count?: number;
  sample_values?: string[];
}

export interface DataPreview {
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  sample_data: Record<string, unknown>[];
  statistics?: Record<string, Record<string, number>>;
  missing_values?: Record<string, number>;
  missing_percentages?: Record<string, number>;
  memory_usage?: number;
  column_info?: ColumnInfo[];
  numeric_columns?: string[];
  categorical_columns?: string[];
  datetime_columns?: string[];
}

export interface SessionInfo {
  session_id: string;
  filename: string;
  created_at: string;
  data_preview: DataPreview;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  type?: 'text' | 'plot' | 'error';
  plotData?: PlotData;
  timestamp: Date;
}

export interface PlotData {
  data: {
    data?: PlotTrace[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
  };
  title?: string;
  type?: string;
  insights?: string;
}

export interface PlotTrace {
  type?: string;
  x?: unknown;
  y?: unknown;
  z?: unknown;
  values?: unknown;
  labels?: unknown;
  name?: string;
  mode?: string;
  geo?: string;
  locationmode?: string;
  locations?: unknown;
  lat?: unknown;
  lon?: unknown;
  hovertemplate?: string;
  customdata?: unknown;
  [key: string]: unknown;
}

export interface QueryResponse {
  response_type: 'plot' | 'statistics' | 'insight' | 'text' | 'error';
  data: Record<string, unknown>;
  message: string;
}

export interface AppSettings {
  theme: 'dark' | 'light' | 'auto';
  fontSize: 'small' | 'medium' | 'large';
  chartTheme: string;
  maxRowsDisplay: number;
  responseStyle: string;
  autoRenderCharts: boolean;
  showDataLabels: boolean;
  cacheResults: boolean;
  showCodeBlocks: boolean;
  autoSuggestions: boolean;
  anonymizeData: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  theme: 'dark',
  fontSize: 'medium',
  chartTheme: 'plotly_dark',
  maxRowsDisplay: 100,
  responseStyle: 'detailed',
  autoRenderCharts: true,
  showDataLabels: false,
  cacheResults: true,
  showCodeBlocks: false,
  autoSuggestions: true,
  anonymizeData: false,
};
