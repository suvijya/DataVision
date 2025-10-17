# 🤖 PyData Assistant - AI-Powered Data Analysis Platform

Transform your CSV data into insights using natural language queries powered by Google's Gemini AI.

## ✨ Features

- 📊 **Smart CSV Analysis** - Upload and get instant insights from your datasets
- 🤖 **AI-Powered Queries** - Ask questions in plain English and get intelligent responses
- 📈 **Interactive Visualizations** - Plotly charts generated automatically from your queries
- 💬 **Conversational Interface** - ChatGPT-like experience for data exploration
- 🔒 **Secure Execution** - Sandboxed code execution with safety restrictions
- 💾 **Session Management** - Persistent analysis sessions with conversation history
- 🌐 **Modern Web Interface** - Responsive design with drag-and-drop file uploads

## 🏗️ Architecture

```
pydatabackend/
├── app/
│   ├── core/
│   │   └── config.py              # Application settings
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── session.py     # API endpoints
│   │       └── schemas/
│   │           └── session.py     # Pydantic models
│   ├── services/
│   │   ├── session_manager.py     # Session lifecycle management
│   │   └── data_analysis.py       # LLM integration & code execution
│   └── main.py                    # FastAPI application
├── frontend/
│   ├── index.html                 # Main UI
│   ├── styles.css                 # Styling
│   └── script.js                  # JavaScript functionality
├── cache/                         # Session data storage
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Google Gemini API key (free from [Google AI Studio](https://makersuite.google.com/app/apikey))

### 2. Installation

```bash
# Clone or navigate to the project directory
cd pydatabackend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
copy .env.example .env

# Edit .env file and add your Gemini API key
# Get your API key from: https://makersuite.google.com/app/apikey
```

**Required Environment Variables:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=True
```

### 4. Run the Application

```bash
python app/main.py
```

The application will start on `http://localhost:8000`

**Available URLs:**
- **Main App**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🎯 Usage Guide

### 1. Upload Your Data
- Drag and drop a CSV file (max 16MB) or click to browse
- Supported format: CSV files with UTF-8 encoding
- The system will automatically analyze your data structure

### 2. Explore Your Data
- **Sample Data Tab**: View the first 5 rows
- **Statistics Tab**: See numeric statistics and missing value analysis
- **Data Info Tab**: Check column types and categorical information

### 3. Ask Questions
Use natural language to analyze your data:

**Example Queries:**
- "Show me a summary of the data"
- "Create a histogram of the age column"
- "What's the correlation between price and sales?"
- "Are there any missing values?"
- "Show sales by region as a bar chart"
- "Create a scatter plot of height vs weight"
- "What are the top 10 customers by revenue?"

## 🔧 API Endpoints

### Session Management
- `POST /api/v1/session/start` - Upload CSV and start session
- `POST /api/v1/session/query` - Submit analysis query
- `GET /api/v1/sessions` - List active sessions
- `GET /api/v1/session/{session_id}` - Get session info
- `DELETE /api/v1/session/{session_id}` - Delete session

### Example API Usage

```bash
# Start a session
curl -X POST "http://localhost:8000/api/v1/session/start" \
  -F "file=@your_data.csv"

# Query the data
curl -X POST "http://localhost:8000/api/v1/session/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "query": "Show me sales by region"
  }'
```

## 🛡️ Security Features

- **Restricted Imports**: Only safe libraries are allowed (pandas, numpy, plotly, etc.)
- **Code Sandboxing**: Generated code runs in a controlled environment
- **Input Validation**: File type and size validation
- **Session Isolation**: Each session operates independently

## 🎨 Frontend Features

- **Modern UI**: Clean, responsive design with gradient backgrounds
- **Drag & Drop**: Intuitive file upload experience
- **Real-time Chat**: Interactive conversation interface
- **Data Visualization**: Integrated Plotly charts
- **Error Handling**: User-friendly error messages and loading states
- **Mobile Responsive**: Works on desktop and mobile devices

## 📊 Supported Analysis Types

### Data Exploration
- Dataset overview and statistics
- Missing value analysis
- Data type information
- Sample data preview

### Visualizations
- Bar charts, line charts, scatter plots
- Histograms and distribution plots
- Correlation matrices
- Custom Plotly visualizations

### Statistical Analysis
- Descriptive statistics
- Correlation analysis
- Grouping and aggregation
- Trend analysis

## 🔧 Development

### Project Structure
```
app/
├── core/           # Core configuration and settings
├── api/            # API routes and schemas
├── services/       # Business logic services
└── main.py         # Application entry point
```

### Key Components
- **FastAPI**: Modern Python web framework
- **Google Gemini**: Advanced AI for code generation
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Pydantic**: Data validation and settings

### Environment Variables
```env
# Required
GEMINI_API_KEY=your_key_here

# Optional (with defaults)
DEBUG=True
MAX_FILE_SIZE=16777216
CACHE_DIR=cache
SESSION_TIMEOUT=86400
LLM_MODEL=gemini-1.5-flash
```

## 🐛 Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY is required" error**
   - Ensure you've set your API key in the `.env` file
   - Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Module import errors**
   - Activate your virtual environment
   - Run `pip install -r requirements.txt`

3. **File upload fails**
   - Check file format (must be CSV)
   - Verify file size (max 16MB)
   - Ensure UTF-8 encoding

4. **Charts not displaying**
   - Check browser console for JavaScript errors
   - Ensure internet connection for Plotly.js CDN

### Performance Tips
- Use smaller datasets for faster processing
- Complex visualizations may take longer to generate
- Sessions are cached for 24 hours by default

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini** for powerful AI capabilities
- **FastAPI** for the excellent web framework
- **Plotly** for beautiful visualizations
- **Pandas** for robust data analysis tools

---

**Ready to explore your data with AI? Upload a CSV file and start asking questions!** 🚀