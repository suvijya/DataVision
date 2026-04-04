'use client';

import { useState, useCallback } from 'react';
import { useSession } from '@/hooks/useSession';
import type { DataPreview as DataPreviewType } from '@/types';
import * as api from '@/lib/api';

const TABS = [
  { id: 'overview', icon: 'fa-eye', label: 'Overview' },
  { id: 'sample', icon: 'fa-table', label: 'Sample Data' },
  { id: 'fulldata', icon: 'fa-database', label: 'Full Data' },
  { id: 'stats', icon: 'fa-chart-bar', label: 'Statistics' },
  { id: 'quality', icon: 'fa-check-circle', label: 'Data Quality' },
  { id: 'columns', icon: 'fa-columns', label: 'Columns' },
  { id: 'statistical', icon: 'fa-flask', label: 'Statistical Analysis' },
  { id: 'insights', icon: 'fa-lightbulb', label: 'AI Insights' },
  { id: 'chartgallery', icon: 'fa-chart-pie', label: 'Chart Gallery' },
];

export default function DataPreview({ data }: { data: DataPreviewType }) {
  const [activeTab, setActiveTab] = useState('sample');

  return (
    <section className="preview-section">
      <div className="section-header">
        <h2><i className="fas fa-eye" /> Data Preview</h2>
        <div className="dataset-info">
          <span><strong>{data.shape[0]}</strong> rows, <strong>{data.shape[1]}</strong> columns</span>
        </div>
      </div>
      <div className="preview-content">
        <div className="preview-sidebar">
          <div className="sidebar-tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                data-tab={tab.id}
                onClick={() => setActiveTab(tab.id)}
              >
                <i className={`fas ${tab.icon}`} />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="preview-main">
          <div className="tab-content">
            {activeTab === 'overview' && <OverviewContent data={data} />}
            {activeTab === 'sample' && <SampleContent data={data} />}
            {activeTab === 'fulldata' && <FullDataContent />}
            {activeTab === 'stats' && <StatsContent data={data} />}
            {activeTab === 'quality' && <QualityContent data={data} />}
            {activeTab === 'columns' && <ColumnsContent data={data} />}
            {activeTab === 'statistical' && <StatisticalAnalysisContent data={data} />}
            {activeTab === 'insights' && <InsightsContent data={data} />}
            {activeTab === 'chartgallery' && <ChartGalleryContent data={data} />}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ========== HELPER ========== */
function truncateText(text: string, maxLen: number): string {
  return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getColumnCategories(data: DataPreviewType) {
  const dtypes = data.dtypes || {};
  const columns = data.columns || [];
  const numericCols = columns.filter(c => dtypes[c] && (dtypes[c].includes('int') || dtypes[c].includes('float')));
  const categoricalCols = columns.filter(c => dtypes[c] && dtypes[c].includes('object'));
  const dateCols = columns.filter(c =>
    dtypes[c] && (dtypes[c].includes('datetime') || c.toLowerCase().includes('date') || c.toLowerCase().includes('time'))
  );
  return { numericCols, categoricalCols, dateCols };
}

/* ========== OVERVIEW TAB ========== */
function OverviewContent({ data }: { data: DataPreviewType }) {
  return (
    <div className="overview-content">
      <div className="stats-grid">
        <div className="stat-card">
          <h4>Dataset Size</h4>
          <div className="value">{data.shape[0].toLocaleString()} × {data.shape[1]}</div>
          <div className="label">rows × columns</div>
        </div>
        <div className="stat-card">
          <h4>Memory Usage</h4>
          <div className="value">{formatBytes(data.memory_usage || 0)}</div>
          <div className="label">total size</div>
        </div>
      </div>
      <p><strong>Columns:</strong> {data.columns.join(', ')}</p>
    </div>
  );
}

/* ========== SAMPLE DATA TAB ========== */
function SampleContent({ data }: { data: DataPreviewType }) {
  if (!data.sample_data || data.sample_data.length === 0) {
    return <p className="text-muted">No sample data available</p>;
  }

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            {data.columns.map((col) => <th key={col}>{col}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.sample_data.map((row, i) => (
            <tr key={i}>
              {data.columns.map((col) => (
                <td key={col}>
                  {truncateText(String(row[col] !== null && row[col] !== undefined ? row[col] : ''), 50)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ========== FULL DATA TAB ========== */
function FullDataContent() {
  const { sessionId } = useSession();
  const [loading, setLoading] = useState(false);
  const [fullData, setFullData] = useState<{ data: Record<string, unknown>[]; columns: string[]; pagination: Record<string, number> } | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async (p: number, ps: number) => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.getSessionData(sessionId, p, ps);
      setFullData(result);
      setPage(p);
      setPageSize(ps);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Auto-load on first render
  if (!fullData && !loading && !error) {
    loadData(1, 100);
  }

  if (loading) {
    return <div className="loading-spinner"><i className="fas fa-spinner fa-spin" /> Loading full dataset...</div>;
  }
  if (error) {
    return <p className="error-text">❌ Error: {error}</p>;
  }
  if (!fullData) return null;

  const { data, columns, pagination } = fullData;
  const totalPages = pagination.total_pages || 1;

  return (
    <div className="full-data-container">
      <div className="pagination-header">
        <div className="pagination-info">
          Showing {pagination.start_row}-{pagination.end_row} of {pagination.total_rows?.toLocaleString() || 0} rows
        </div>
        <div className="pagination-controls">
          <button className="page-btn" disabled={page === 1} onClick={() => loadData(1, pageSize)}>
            <i className="fas fa-angle-double-left" />
          </button>
          <button className="page-btn" disabled={page === 1} onClick={() => loadData(page - 1, pageSize)}>
            <i className="fas fa-angle-left" /> Prev
          </button>
          <span className="page-number">Page {page} of {totalPages}</span>
          <button className="page-btn" disabled={page === totalPages} onClick={() => loadData(page + 1, pageSize)}>
            Next <i className="fas fa-angle-right" />
          </button>
          <button className="page-btn" disabled={page === totalPages} onClick={() => loadData(totalPages, pageSize)}>
            <i className="fas fa-angle-double-right" />
          </button>
        </div>
        <div className="page-size-selector">
          <label>Rows per page:</label>
          <select value={pageSize} onChange={(e) => loadData(1, parseInt(e.target.value))}>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
            <option value={500}>500</option>
          </select>
        </div>
      </div>
      <div className="table-container scrollable-table">
        <table className="data-table full-data-table">
          <thead>
            <tr>
              <th className="row-index">#</th>
              {columns.map((col) => <th key={col}>{col}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => {
              const rowNum = pagination.start_row + idx;
              return (
                <tr key={idx}>
                  <td className="row-index">{rowNum}</td>
                  {columns.map((col) => (
                    <td key={col}>{truncateText(String(row[col] !== null && row[col] !== undefined ? row[col] : ''), 100)}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ========== STATISTICS TAB ========== */
function StatsContent({ data }: { data: DataPreviewType }) {
  const totalMissing = data.missing_values ? Object.values(data.missing_values).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="stats-container">
      <div className="stats-grid">
        <div className="stat-card">
          <h4>Total Rows</h4>
          <div className="value">{data.shape[0].toLocaleString()}</div>
        </div>
        <div className="stat-card">
          <h4>Total Columns</h4>
          <div className="value">{data.shape[1]}</div>
        </div>
        {data.missing_values && (
          <div className="stat-card">
            <h4>Missing Values</h4>
            <div className="value">{totalMissing.toLocaleString()}</div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ========== DATA QUALITY TAB ========== */
function QualityContent({ data }: { data: DataPreviewType }) {
  if (!data.missing_values) {
    return <p>No data quality information available.</p>;
  }

  return (
    <div className="quality-content">
      <h4>Missing Values by Column</h4>
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>
          </thead>
          <tbody>
            {Object.entries(data.missing_values).map(([col, missing]) => {
              const pct = data.missing_percentages ? data.missing_percentages[col] || 0 : 0;
              return (
                <tr key={col}>
                  <td><strong>{col}</strong></td>
                  <td>{missing}</td>
                  <td>{pct.toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ========== COLUMNS INFO TAB ========== */
function ColumnsContent({ data }: { data: DataPreviewType }) {
  return (
    <div className="columns-content">
      <h4>Column Information</h4>
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr><th>Column</th><th>Data Type</th></tr>
          </thead>
          <tbody>
            {data.columns.map((col) => (
              <tr key={col}>
                <td><strong>{col}</strong></td>
                <td>{data.dtypes ? data.dtypes[col] || 'unknown' : 'unknown'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ========== INSIGHTS TAB ========== */
function InsightsContent({ data }: { data: DataPreviewType }) {
  return (
    <div className="insights-content">
      <h4>📊 Dataset Insights</h4>
      <ul>
        <li>Your dataset contains <strong>{data.shape[0]} rows</strong> and <strong>{data.shape[1]} columns</strong></li>
        <li>Columns: {data.columns.join(', ')}</li>
        <li>Upload completed successfully and data is ready for analysis</li>
      </ul>
      <p><em>Use the chat interface to ask specific questions about your data!</em></p>
    </div>
  );
}

/* ========== STATISTICAL ANALYSIS TAB ========== */
function StatisticalAnalysisContent({ data }: { data: DataPreviewType }) {
  const { sendQuery } = useSession();
  const { numericCols, categoricalCols } = getColumnCategories(data);

  const execute = (query: string) => sendQuery(query);

  return (
    <div className="statistical-analysis-content">
      <h3><i className="fas fa-flask" /> Professional Statistical Analysis Suite</h3>
      <p className="stats-intro">🎉 <strong>11 Advanced Statistical Tools</strong> — Click any analysis below to run it on your data!</p>

      {/* Hypothesis Testing */}
      <div className="stats-category">
        <h4><i className="fas fa-vial" /> Hypothesis Testing</h4>
        <div className="stats-grid">
          {numericCols.length > 0 && (
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-chart-bar" /></div>
              <h5>Normality Tests</h5>
              <p>Test if data follows normal distribution</p>
              <div className="tool-actions" style={{ display: 'flex', gap: '8px' }}>
                <button className="stat-btn" style={{ flex: 1 }} onClick={() => execute(`Test if ${numericCols[0]} is normally distributed using Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov tests. Show test statistics and p-values in TEXT FORMAT ONLY`)}>
                  <i className="fas fa-list" /> Analyze
                </button>
                <button className="stat-btn stat-btn-viz" style={{ flex: 1 }} onClick={() => execute(`Create a histogram with normal distribution overlay for ${numericCols[0]} to visualize normality`)}>
                  <i className="fas fa-chart-bar" /> Visualize
                </button>
              </div>
              <div className="tool-info"><small><strong>Analyze:</strong> Shapiro-Wilk, Anderson-Darling, KS | <strong>Visualize:</strong> Histogram + normal curve</small></div>
            </div>
          )}
          {numericCols.length >= 2 && (
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-balance-scale" /></div>
              <h5>T-Test (Compare Groups)</h5>
              <p>Compare means between two groups with effect size</p>
              <div className="tool-actions">
                <button className="stat-btn" onClick={() => execute(`Perform independent t-test comparing ${numericCols[0]} and ${numericCols[1]} including Cohen's d effect size`)}>
                  Compare {numericCols[0]} vs {numericCols[1]}
                </button>
              </div>
              <div className="tool-info"><small><strong>Includes:</strong> t-statistic, p-value, Cohen&apos;s d, confidence interval</small></div>
            </div>
          )}
          {categoricalCols.length > 0 && numericCols.length > 0 && (
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-layer-group" /></div>
              <h5>ANOVA (Multiple Groups)</h5>
              <p>Compare means across 3+ groups</p>
              <div className="tool-actions">
                <button className="stat-btn" onClick={() => execute(`Perform one-way ANOVA comparing ${numericCols[0]} across different ${categoricalCols[0]} groups with eta-squared effect size`)}>
                  {numericCols[0]} by {categoricalCols[0]}
                </button>
              </div>
              <div className="tool-info"><small><strong>Includes:</strong> F-statistic, p-value, eta-squared</small></div>
            </div>
          )}
          {numericCols.length >= 2 && (
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-link" /></div>
              <h5>Correlation Analysis</h5>
              <p>Test correlation significance (3 methods)</p>
              <div className="tool-actions" style={{ display: 'flex', gap: '8px' }}>
                <button className="stat-btn" style={{ flex: 1 }} onClick={() => execute(`Test correlation between ${numericCols[0]} and ${numericCols[1]} using Pearson, Spearman, and Kendall methods. Show coefficients and p-values in TEXT FORMAT ONLY`)}>
                  <i className="fas fa-list" /> Analyze
                </button>
                <button className="stat-btn stat-btn-viz" style={{ flex: 1 }} onClick={() => execute(`Create a scatter plot showing correlation between ${numericCols[0]} and ${numericCols[1]} with trend line and Pearson correlation coefficient`)}>
                  <i className="fas fa-chart-line" /> Visualize
                </button>
              </div>
              <div className="tool-info"><small><strong>Analyze:</strong> Pearson, Spearman, Kendall | <strong>Visualize:</strong> Scatter plot with trend line</small></div>
            </div>
          )}
        </div>
      </div>

      {/* Outlier Detection */}
      {numericCols.length > 0 && (
        <div className="stats-category">
          <h4><i className="fas fa-bullseye" /> Outlier Detection</h4>
          <div className="stats-grid">
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-search" /></div>
              <h5>Advanced Outlier Detection</h5>
              <p>Detect anomalies using 4 different methods</p>
              <div className="tool-actions" style={{ display: 'flex', gap: '8px' }}>
                <button className="stat-btn" style={{ flex: 1 }} onClick={() => execute(`Detect outliers in ${numericCols[0]} using all methods: IQR, Z-score, Modified Z-score (MAD), and Isolation Forest. Show counts and threshold values in TEXT FORMAT ONLY`)}>
                  <i className="fas fa-list" /> Analyze
                </button>
                <button className="stat-btn stat-btn-viz" style={{ flex: 1 }} onClick={() => execute(`Create a box plot for ${numericCols[0]} to visualize outliers with IQR method. Highlight outliers in red`)}>
                  <i className="fas fa-chart-bar" /> Visualize
                </button>
              </div>
              <div className="tool-info"><small><strong>Analyze:</strong> IQR, Z-score, MAD, Isolation Forest | <strong>Visualize:</strong> Box plot with outliers</small></div>
            </div>
          </div>
        </div>
      )}

      {/* Regression Analysis */}
      {numericCols.length >= 2 && (
        <div className="stats-category">
          <h4><i className="fas fa-chart-line" /> Regression Analysis</h4>
          <div className="stats-grid">
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-chart-line" /></div>
              <h5>Linear Regression</h5>
              <p>Model linear relationships between variables</p>
              <div className="tool-actions" style={{ display: 'flex', gap: '8px' }}>
                <button className="stat-btn" style={{ flex: 1 }} onClick={() => execute(`Perform linear regression with ${numericCols[0]} as predictor and ${numericCols[1]} as target. Show R², RMSE, coefficients. Show first 10 predictions in TEXT FORMAT ONLY`)}>
                  <i className="fas fa-list" /> Analyze
                </button>
                <button className="stat-btn stat-btn-viz" style={{ flex: 1 }} onClick={() => execute(`Create a scatter plot showing ${numericCols[1]} vs ${numericCols[0]} with linear regression line. Include regression equation and R² score`)}>
                  <i className="fas fa-chart-line" /> Visualize
                </button>
              </div>
              <div className="tool-info"><small><strong>Current:</strong> {numericCols[0]} → {numericCols[1]}</small></div>
            </div>
          </div>
        </div>
      )}

      {/* Distribution Fitting */}
      {numericCols.length > 0 && (
        <div className="stats-category">
          <h4><i className="fas fa-chart-area" /> Distribution Fitting</h4>
          <div className="stats-grid">
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-chart-area" /></div>
              <h5>Fit Probability Distributions</h5>
              <p>Test which distribution best fits your data</p>
              <div className="tool-actions">
                <button className="stat-btn" onClick={() => execute(`Fit various probability distributions (normal, exponential, gamma, beta, lognormal, weibull, etc.) to ${numericCols[0]} and find the best fit`)}>
                  Fit {numericCols[0]}
                </button>
              </div>
              <div className="tool-info"><small><strong>13 Distributions:</strong> normal, exponential, gamma, beta, lognormal, weibull, uniform, chi2, t, f, pareto, logistic, gumbel</small></div>
            </div>
          </div>
        </div>
      )}

      {/* Summary Statistics */}
      {numericCols.length > 0 && (
        <div className="stats-category">
          <h4><i className="fas fa-calculator" /> Summary Statistics</h4>
          <div className="stats-grid">
            <div className="stat-tool-card">
              <div className="tool-icon"><i className="fas fa-list-ol" /></div>
              <h5>Comprehensive Statistics</h5>
              <p>Get detailed descriptive statistics</p>
              <div className="tool-actions">
                <button className="stat-btn" onClick={() => execute(`Get comprehensive statistics for ${numericCols[0]} including mean, median, mode, std, variance, skewness, kurtosis, quartiles, and range`)}>
                  Analyze {numericCols[0]}
                </button>
              </div>
              <div className="tool-info"><small><strong>Metrics:</strong> Central tendency, dispersion, shape, position</small></div>
            </div>
          </div>
        </div>
      )}

      {/* Custom Analysis */}
      <div className="stats-category">
        <h4><i className="fas fa-keyboard" /> Custom Analysis</h4>
        <div className="custom-stats-input">
          <p>Or type your own statistical analysis query in the chat below. Examples:</p>
          <ul className="stats-examples">
            {numericCols.length > 0 && (
              <>
                <li>Test if {numericCols[0]} follows a normal distribution</li>
                <li>Detect outliers in {numericCols[0]} using Isolation Forest</li>
              </>
            )}
            {numericCols.length >= 2 && (
              <>
                <li>Perform regression with {numericCols[0]} predicting {numericCols[1]}</li>
                <li>Test correlation between {numericCols[0]} and {numericCols[1]}</li>
              </>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* ========== CHART GALLERY TAB ========== */
function ChartGalleryContent({ data }: { data: DataPreviewType }) {
  const { sendQuery } = useSession();
  const { numericCols, categoricalCols } = getColumnCategories(data);

  const chartCategories = [
    {
      category: 'Distribution & Comparison',
      charts: [
        { name: 'Bar Chart', icon: 'fa-chart-bar', description: 'Compare categories or values across groups' },
        { name: 'Histogram', icon: 'fa-chart-area', description: 'Visualize data distribution and frequency' },
        { name: 'Box Plot', icon: 'fa-box', description: 'Display quartiles, median, and outliers' },
        { name: 'Violin Plot', icon: 'fa-music', description: 'Combine box plot with kernel density' },
      ],
    },
    {
      category: 'Relationships & Correlations',
      charts: [
        { name: 'Scatter Plot', icon: 'fa-braille', description: 'Show relationship between two variables' },
        { name: 'Bubble Chart', icon: 'fa-circle', description: 'Scatter plot with size dimension' },
        { name: 'Heatmap', icon: 'fa-th', description: 'Display correlations in a color matrix' },
        { name: 'Line Chart', icon: 'fa-chart-line', description: 'Show trends over time or sequences' },
      ],
    },
    {
      category: 'Proportions & Parts',
      charts: [
        { name: 'Pie Chart', icon: 'fa-chart-pie', description: 'Show percentage breakdown of categories' },
        { name: 'Donut Chart', icon: 'fa-circle-notch', description: 'Pie chart with a center hole' },
        { name: 'Sunburst', icon: 'fa-sun', description: 'Hierarchical data in concentric circles' },
        { name: 'Treemap', icon: 'fa-th-large', description: 'Nested rectangles for hierarchical data' },
      ],
    },
    {
      category: 'Advanced & Specialized',
      charts: [
        { name: '3D Scatter', icon: 'fa-cube', description: 'Three-dimensional scatter plot' },
        { name: 'Funnel Chart', icon: 'fa-filter', description: 'Visualize progressive reduction in stages' },
        { name: 'Waterfall', icon: 'fa-water', description: 'Show cumulative effect of sequential values' },
        { name: 'Density Plot', icon: 'fa-wave-square', description: 'Smooth distribution estimate' },
      ],
    },
  ];

  const generatePrompt = (chartName: string): string => {
    const num = numericCols;
    const cat = categoricalCols;
    switch (chartName) {
      case 'Bar Chart': return num.length > 0 && cat.length > 0 ? `Create a bar chart of ${num[0]} by ${cat[0]}` : `Create a bar chart of the data`;
      case 'Histogram': return num.length > 0 ? `Create a histogram of ${num[0]} distribution` : `Create a histogram`;
      case 'Box Plot': return num.length > 0 && cat.length > 0 ? `Create a box plot of ${num[0]} by ${cat[0]}` : `Create a box plot`;
      case 'Violin Plot': return num.length > 0 && cat.length > 0 ? `Create a violin plot of ${num[0]} by ${cat[0]}` : `Create a violin plot`;
      case 'Scatter Plot': return num.length >= 2 ? `Create a scatter plot of ${num[0]} vs ${num[1]}` : `Create a scatter plot`;
      case 'Bubble Chart': return num.length >= 3 ? `Create a bubble chart: ${num[0]} vs ${num[1]}, size = ${num[2]}` : `Create a bubble chart`;
      case 'Heatmap': return `Create a correlation heatmap of all numeric columns`;
      case 'Line Chart': return num.length >= 2 ? `Create a line chart of ${num[0]} over ${num[1]}` : `Create a line chart`;
      case 'Pie Chart': return cat.length > 0 && num.length > 0 ? `Create a pie chart of ${num[0]} by ${cat[0]}` : `Create a pie chart`;
      case 'Donut Chart': return cat.length > 0 && num.length > 0 ? `Create a donut chart showing ${num[0]} by ${cat[0]}` : `Create a donut chart`;
      case 'Sunburst': return cat.length >= 2 && num.length > 0 ? `Create a sunburst chart of ${num[0]} by ${cat[0]} and ${cat[1]}` : `Create a sunburst chart`;
      case 'Treemap': return cat.length > 0 && num.length > 0 ? `Create a treemap of ${num[0]} by ${cat[0]}` : `Create a treemap`;
      case '3D Scatter': return num.length >= 3 ? `Create a 3D scatter plot of ${num[0]} vs ${num[1]} vs ${num[2]}` : `Create a 3D scatter`;
      case 'Funnel Chart': return cat.length > 0 && num.length > 0 ? `Create a funnel chart of ${num[0]} by ${cat[0]}` : `Create a funnel chart`;
      case 'Waterfall': return num.length > 0 && cat.length > 0 ? `Create a waterfall chart of ${num[0]} by ${cat[0]}` : `Create a waterfall chart`;
      case 'Density Plot': return num.length > 0 ? `Create a density plot of ${num[0]}` : `Create a density plot`;
      default: return `Create a ${chartName.toLowerCase()}`;
    }
  };

  return (
    <div className="chart-gallery-content">
      <h3><i className="fas fa-palette" /> Available Visualization Types</h3>
      <p className="gallery-intro">Choose from 25+ chart types to analyze and visualize your data. Smart suggestions based on your dataset!</p>

      {(numericCols.length > 0 || categoricalCols.length > 0) && (
        <div className="dataset-info-banner">
          <i className="fas fa-database" /> <strong>Your Dataset:</strong>{' '}
          {numericCols.length > 0 && <span className="hint-badge">📊 {numericCols.length} numeric column{numericCols.length > 1 ? 's' : ''}</span>}
          {' '}
          {categoricalCols.length > 0 && <span className="hint-badge">🏷️ {categoricalCols.length} categorical column{categoricalCols.length > 1 ? 's' : ''}</span>}
        </div>
      )}

      {chartCategories.map((cat) => (
        <div key={cat.category} className="chart-category">
          <h4><i className="fas fa-folder-open" /> {cat.category}</h4>
          <div className="charts-grid">
            {cat.charts.map((chart) => (
              <div className="chart-card" key={chart.name}>
                <div className="chart-icon"><i className={`fas ${chart.icon}`} /></div>
                <div className="chart-info">
                  <h5>{chart.name}</h5>
                  <p className="chart-description">{chart.description}</p>
                  <button className="stat-btn" onClick={() => sendQuery(generatePrompt(chart.name))}>
                    Create {chart.name}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
