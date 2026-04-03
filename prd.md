# 📊 DataVision Assistant — Product Requirements Document (PRD)

> **Version:** 2.1  
> **Last Updated:** April 2026  
> **Status:** Active Development  
> **Product Owner:** Suvijya Arya

---

## 1. Executive Summary

**DataVision Assistant** is an AI-powered, web-based data analysis platform that lets users upload CSV datasets and explore them through natural language queries. It democratizes data science by eliminating the need to write code — users can ask questions in plain English and receive charts, statistics, and tables in real-time.

The platform combines a **FastAPI Python backend** with a **vanilla HTML/CSS/JS frontend**, a **Google Gemini LLM** for intelligent code generation, sandboxed Python execution for security, and **Plotly.js** for interactive visualizations.

---

## 2. Vision & Goals

### Product Vision
> *"Turn anyone into a data analyst — no code required."*

### Primary Goals
| Goal | Description |
|---|---|
| **Accessibility** | Let non-technical users perform professional-grade data analysis |
| **Speed** | Deliver meaningful insights within seconds of uploading a dataset |
| **Safety** | Execute LLM-generated code in a fully sandboxed environment |
| **Intelligence** | Leverage frontier AI models to understand ambiguous natural language queries |
| **Extensibility** | Provide a clean REST API for third-party integration |

---

## 3. Target Users

| Persona | Description | Key Needs |
|---|---|---|
| **Data Analysts** | Business or academic analysts who regularly work with CSV exports | Fast exploratory analysis, charting, statistical tests |
| **Business Users** | Non-technical managers or marketers reviewing campaign/sales data | Simple natural language Q&A, clear visualizations |
| **Researchers** | Academic or scientific users running statistical hypothesis tests | Normality tests, regression, p-values, effect sizes |
| **Students** | Learning data science fundamentals | Guided exploration, interactive charts, sample queries |
| **Developers** | Integrating analytics into their own products | Clean REST API, session management, documented endpoints |

---

## 4. Current System Architecture

### 4.1 Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python 3.8+) with Uvicorn ASGI |
| **LLM** | Google Gemini (`gemini-2.0-flash-lite-preview`) |
| **Data Processing** | Pandas 2.1+, NumPy 1.25+ |
| **Statistical Analysis** | SciPy 1.11+, scikit-learn 1.3+, statsmodels 0.14+, Prophet |
| **Visualization** | Plotly 5.17+ (backend), Plotly.js 2.27 (frontend) |
| **Frontend** | Vanilla HTML5, CSS3 (CSS Variables, glassmorphism), JavaScript (ES6+) |
| **Configuration** | Pydantic Settings, python-dotenv |
| **Session Storage** | File-based cache (`/cache` directory) |
| **Testing** | pytest, pytest-asyncio, httpx |

### 4.2 File Structure (Simplified)

```
pydatabackend/
├── app/
│   ├── core/simple_config.py       # Env vars & settings
│   ├── api/v1/endpoints/
│   │   ├── session.py              # Upload & query endpoints
│   │   └── statistical_analysis.py # Dedicated stats endpoints
│   ├── services/
│   │   ├── data_analysis.py        # LLM integration, code generation, sandbox
│   │   ├── statistical_analysis.py # Statistical computations
│   │   └── session_manager.py      # Session lifecycle
│   └── main.py                     # FastAPI app entry point
├── frontend/
│   ├── index.html                  # Single-page application
│   ├── styles.css                  # Dark-theme design system
│   └── script.js                   # All UI logic (~3,100 lines)
└── cache/                          # Per-session data storage
```

### 4.3 Data Flow

```
User uploads CSV
      │
      ▼
FastAPI /session/start → Pandas parsing → Session stored in /cache
      │
      ▼
User submits a natural language query
      │
      ▼
data_analysis.py → Build LLM prompt (with column info + instructions)
      │
      ▼
Google Gemini API → Generates Python code (pandas, plotly, stats)
      │
      ▼
Sandboxed exec() → Code runs with restricted globals
      │
      ▼
Response (JSON): type = [plot | statistics | insight | text | error]
      │
      ▼
Frontend script.js → Renders: Plotly chart OR formatted HTML table/text
```

---

## 5. Feature Inventory (Current State)

### 5.1 Core Features

#### ✅ CSV Upload & Session Management
- Drag-and-drop or click-to-browse upload (max 16 MB)
- Auto-column type detection (numeric, categorical, datetime)
- File validation (CSV only, UTF-8)
- Session-based architecture — each user gets an isolated session ID
- Session expiry: 24 hours (configurable via `SESSION_TIMEOUT`)
- Session info displayed in header (ID + duration timer)

#### ✅ AI-Powered Query Engine
- Natural language query input (up to 1000 characters)
- LLM generates Python code, which is executed in a sandbox
- Sandboxed environment allows: `pandas`, `numpy`, `plotly`, `scipy`, `statsmodels`, `sklearn`, `math`, `statistics`, `itertools`
- LLM prompt system includes dataset column info, sample values, and explicit formatting rules
- Response types: `plot`, `statistics`, `insight`, `text`, `error`

#### ✅ Data Preview Panel (6 Tabs)
| Tab | Contents |
|---|---|
| **Overview** | Row/column counts, dtypes, memory usage |
| **Sample Data** | First rows with horizontal scrolling |
| **Statistics** | `df.describe()` for numeric columns |
| **Data Quality** | Missing value counts and percentages |
| **Columns** | Per-column detailed analysis |
| **AI Insights** | Auto-generated dataset observations |

#### ✅ Interactive Visualizations (via Plotly.js)
Supported chart types generated by the LLM:
- Bar charts, grouped bar charts
- Line charts & area charts
- Scatter plots (with OLS trendlines)
- Histograms & distribution plots
- Box plots & violin plots
- Pie / donut charts
- Correlation heatmaps
- 3D scatter plots
- Bubble charts
- Sunburst / treemap charts
- Geographic choropleths

#### ✅ Tabular Data Rendering
- LLM uses injected `to_markdown(df)` helper to generate Markdown tables
- Frontend parser converts Markdown table syntax (`|...|...|`) to styled HTML `<table>` elements
- Responsive horizontal scrolling for wide tables
- Zebra striping and row hover effects

#### ✅ Statistical Analysis Suite (Dedicated Endpoints)
Accessed via both the chat interface and dedicated UI panel:

| Category | Tests Available |
|---|---|
| **Normality** | Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, KS |
| **Hypothesis** | Independent t-test, Welch's t-test, Paired t-test |
| **Comparison** | One-way ANOVA (with eta-squared) |
| **Independence** | Chi-Square (with Cramér's V) |
| **Correlation** | Pearson, Spearman, Kendall Tau |
| **Outlier Detection** | IQR, Z-Score, Modified Z-Score (MAD), Isolation Forest |
| **Regression** | Linear, Polynomial (any degree), Logistic |
| **Distribution Fitting** | Normal, Exponential, Gamma, Log-normal, Weibull |
| **Time Series** | Augmented Dickey-Fuller (stationarity), Granger Causality |
| **Summary Stats** | Mean, median, mode, std, IQR, skewness, kurtosis, CV |

#### ✅ Dual-Mode Statistical Interface
Each statistical tool card has two buttons:
- **📋 Analyze** → Text-based output (~2s, no chart overhead)
- **📊 Visualize** → Plotly chart output (~5s, interactive)

#### ✅ Chat Interface
- Conversational UI with user / assistant message bubbles
- Message history within the session
- Quick-action buttons (Describe, Missing Values, Correlations, Outliers)
- Smart Suggestions sidebar (Data Exploration / Visualization / Statistical Analysis categories)
- Query history panel with search
- Keyboard shortcuts (Enter to send, Escape to close modals, Ctrl+/ for help)
- Character counter (up to 1,000 chars)

#### ✅ Export System
- Export data as CSV
- Export conversation history
- Export generated charts
- Multiple format support: PDF, HTML, JSON, CSV

#### ✅ Settings Panel
- Theme selection (Light / Dark / Auto)
- Font size control
- Chart theme preference
- Max rows display limit
- Auto-render charts toggle
- Show code blocks toggle
- Auto-suggestions toggle
- Data anonymization (experimental)

#### ✅ Security
- Sandboxed `exec()` with explicit whitelist of safe globals
- Blocked built-ins: `open`, `eval`, `exec`, `import`, `__import__`, OS-level calls
- File type validation on upload
- 16 MB max file size
- CORS middleware with configurable origins
- Session isolation — each session has separate DataFrame in memory

---

## 6. API Reference

### Session Endpoints (`/api/v1/session`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/start` | Upload CSV, create session, return session ID + column overview |
| `POST` | `/query` | Submit NL query, return analysis result |
| `GET` | `/{session_id}` | Get session metadata |
| `GET` | `/{session_id}/data` | Paginated raw data access |
| `DELETE` | `/{session_id}` | Delete session and cache |
| `GET` | `/` (list) | List all active sessions |

### Statistical Analysis Endpoints (`/api/v1/statistical-analysis`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/normality-test` | Run 1-4 normality tests on a column |
| `POST` | `/t-test` | Independent, paired, or Welch's t-test |
| `POST` | `/anova` | One-way ANOVA across categories |
| `POST` | `/chi-square` | Independence test between categorical columns |
| `POST` | `/correlation-test` | Pearson, Spearman, or Kendall correlation |
| `POST` | `/outlier-detection` | IQR, Z-score, MAD, or Isolation Forest |
| `POST` | `/regression` | Linear, Polynomial, or Logistic regression |
| `POST` | `/distribution-fit` | Fit & rank probability distributions |
| `POST` | `/stationarity-test` | Augmented Dickey-Fuller test |
| `POST` | `/granger-causality` | Granger causality for multiple lags |
| `POST` | `/summary-statistics` | Full descriptive statistics for a column |

### System Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve frontend HTML |
| `GET` | `/health` | Health check (status, version) |
| `GET` | `/docs` | Swagger UI (DEBUG mode only) |

---

## 7. Known Issues & Bugs (Current)

| Issue | Severity | Status |
|---|---|---|
| **Markdown table rendering** — LLM correctly generates pipe-delimited tables, but the frontend parser fails to convert them to HTML `<table>` elements in certain environments | 🔴 High | In Progress |
| **Browser caching** — Previous versions of `script.js` may be cached, causing stale behavior; mitigated with `?v=2.1` query param | 🟡 Medium | Mitigated |
| **Binary-encoded Plotly data** — Some NumPy arrays are serialized by Plotly as `{bdata, dtype}` objects; a `decodeBinaryData()` function handles this but may miss edge cases | 🟡 Medium | Partial Fix |
| **120s query timeout** — Complex analyses can hit the frontend abort timeout before the backend responds | 🟡 Medium | Known |
| **Geographic charts** — Choropleth and `scattergeo` traces require additional layout configuration to render correctly | 🟡 Medium | Partial Fix |
| **Session file-based storage** — Cache uses the filesystem; not suitable for multi-instance or distributed deployments | 🟠 Low (scalability) | Deferred |

---

## 8. Planned Features & Roadmap

### Phase 1 — Stability & Polish (Near-Term)
- [ ] **Fix Markdown table rendering** — Ensure HTML tables are always generated for pipe-delimited LLM output
- [ ] **Streaming responses** — Use Server-Sent Events (SSE) to stream LLM output tokens in real-time (reduce perceived latency)
- [ ] **Better error messages** — Categorize and surface LLM / sandbox errors with actionable user guidance
- [ ] **Table export** — Add "Download as CSV" and "Copy to clipboard" buttons on rendered data tables
- [ ] **Auto hard-refresh** — Replace `?v=2.1` cache-busting with a build-time hash

### Phase 2 — Expanded Analysis (Medium-Term)
- [ ] **Multi-file support** — Allow users to upload and join multiple CSV files
- [ ] **Excel (.xlsx) support** — Parse Excel workbooks (multiple sheets)
- [ ] **Data cleaning tools** — UI for dropping nulls, renaming columns, type casting
- [ ] **ARIMA / Prophet forecasting** — Time series forecasting with confidence intervals
- [ ] **Bayesian A/B testing** — Compare variants with posterior distributions
- [ ] **Bootstrap confidence intervals** — Non-parametric uncertainty estimation
- [ ] **Power analysis** — Sample size calculator for experiment design

### Phase 3 — Platform & Scale (Longer-Term)
- [ ] **User authentication** — Login / signup to persist sessions and history across devices
- [ ] **Database-backed sessions** — Replace file cache with PostgreSQL or Redis
- [ ] **Shareable reports** — Generate a static, shareable HTML snapshot of an analysis session
- [ ] **Dashboard builder** — Drag-and-drop layout for pinning favorite charts
- [ ] **Data connectors** — Direct connection to Google Sheets, S3, Postgres
- [ ] **Multi-language LLM support** — Queries in non-English languages
- [ ] **Team collaboration** — Shared sessions, comments, annotations

---

## 9. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | Query response time < 10s for standard analyses; < 30s for complex statistical tests |
| **Reliability** | Sandbox failures return a graceful error; no unhandled exceptions exposed to users |
| **Security** | No access to filesystem, network, environment variables, or OS from within the sandbox |
| **Scalability** | Architecture must support horizontal scaling (session state must be externalizable) |
| **Accessibility** | Keyboard-navigable UI, ARIA labels on interactive elements, responsive down to 480px |
| **Browser Support** | Chrome 90+, Firefox 88+, Safari 14+, Edge 90+ |
| **File Limits** | Max upload size: 16 MB; max rows: no hard limit (performance degrades on >500k rows) |

---

## 10. Environment Configuration

| Variable | Default | Required | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | — | ✅ Yes | Google Gemini API key |
| `LLM_MODEL` | `gemini-1.5-flash` | No | Gemini model name |
| `DEBUG` | `True` | No | Enable Swagger UI and verbose logging |
| `MAX_FILE_SIZE` | `16777216` | No | Max CSV upload size in bytes |
| `CACHE_DIR` | `cache` | No | Directory for session data |
| `SESSION_TIMEOUT` | `86400` | No | Session TTL in seconds (24h) |

---

## 11. Development & Operations

### Starting the Server
```powershell
# Activate venv and run the dev script
.\dev.ps1

# Or manually:
.\venv\Scripts\activate
python start_server.py
```

### Key URLs
| URL | Description |
|---|---|
| `http://localhost:8000/` | Main application |
| `http://localhost:8000/docs` | Swagger UI (DEBUG=True only) |
| `http://localhost:8000/health` | Health check |

### Running Tests
```bash
pytest tests/ -v
```

---

## 12. Open Questions

| # | Question | Priority |
|---|---|---|
| 1 | Should sessions be persisted server-side with a database, or remain file-based? | 🔴 High |
| 2 | Is user authentication in scope for the next release? | 🔴 High |
| 3 | What is the acceptable maximum dataset size (rows × columns)? | 🟡 Medium |
| 4 | Should the LLM model be selectable by the user from the Settings panel? | 🟡 Medium |
| 5 | Is there a monetization or SaaS model planned? | 🟠 Low |
| 6 | Should the tool support non-CSV formats (Parquet, Excel, JSON) in the near term? | 🟠 Low |

---

*This document was auto-generated by Antigravity AI from a full codebase analysis.*  
*Last analysis run: April 4, 2026.*
