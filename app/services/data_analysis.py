"""
Data analysis service with LLM integration.
Handles data previews, query processing, and code execution.
"""

import os
import json
import logging
import time
import ast
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from io import StringIO
import sys
from contextlib import redirect_stdout, redirect_stderr

import google.generativeai as genai
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import silhouette_score

from app.core.simple_config import settings
from app.services.session_manager import Session

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel(settings.LLM_MODEL)


def convert_numpy_types(obj):
    """Convert NumPy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass 
class QueryResponseData:
    """Standardized response data for different query types."""
    response_type: str  # 'insight', 'plot', 'statistics', 'text', 'error'
    data: Any


def get_data_preview(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Generate comprehensive data preview for a DataFrame.
    
    Args:
        df: Pandas DataFrame to analyze
        filename: Original filename
        
    Returns:
        Dictionary containing data preview and metadata
    """
    try:
        # Basic info
        shape = list(df.shape)
        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        
        # Sample data (first 5 rows)
        sample_data = df.head(5).to_dict('records')
        
        # Basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        basic_stats = {}
        
        if numeric_columns:
            stats_df = df[numeric_columns].describe()
            basic_stats = {
                col: {
                    'mean': stats_df.loc['mean', col],
                    'std': stats_df.loc['std', col],
                    'min': stats_df.loc['min', col],
                    'max': stats_df.loc['max', col],
                    'median': stats_df.loc['50%', col]
                }
                for col in numeric_columns
            }
        
        # Categorical columns info
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        categorical_info = {}
        
        for col in categorical_columns:
            unique_count = df[col].nunique()
            categorical_info[col] = {
                'unique_values': unique_count,
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'sample_values': df[col].unique()[:10].tolist()  # First 10 unique values
            }
        
        preview_data = {
            'shape': shape,
            'columns': columns,
            'dtypes': dtypes,
            'sample_data': sample_data,
            'missing_values': missing_values,
            'missing_percentages': missing_percentages,
            'numeric_stats': basic_stats,
            'categorical_info': categorical_info,
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Convert all NumPy types to JSON-serializable types
        return convert_numpy_types(preview_data)
        
    except Exception as e:
        logger.error(f"Error generating data preview: {e}", exc_info=True)
        error_data = {
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_data': [],
            'missing_values': {},
            'error': str(e)
        }
        return convert_numpy_types(error_data)


async def process_query_with_llm(session: Session, query: str) -> Tuple[QueryResponseData, str, float]:
    """
    Process a natural language query using LLM and execute generated code.
    
    Args:
        session: Active session with data
        query: Natural language query
        
    Returns:
        Tuple of (response_data, message, execution_time)
    """
    start_time = time.time()
    
    try:
        # Get data context
        df = session.dataframe
        data_context = _get_data_context(df)
        
        # Get conversation context
        conversation_context = session.get_conversation_context()
        
        # Generate analysis prompt
        prompt = _generate_analysis_prompt(query, data_context, conversation_context)
        
        # Call LLM
        response = model.generate_content(prompt)
        llm_response = response.text
        
        # Extract and execute code
        code_blocks = _extract_code_blocks(llm_response)
        
        if code_blocks:
            # Execute the code
            result = _execute_code_safely(code_blocks[0], df)
            
            if result['success']:
                response_data = _process_execution_result(result, query)
                message = result.get('output', 'Analysis completed successfully')
            else:
                response_data = QueryResponseData(
                    response_type='error',
                    data={
                        'response_type': 'error',
                        'error': result['error'],
                        'message': 'Code execution failed',
                        'error_type': 'execution_error'
                    }
                )
                message = f"Execution failed: {result['error']}"
        else:
            # No code to execute, return text response
            response_data = QueryResponseData(
                response_type='text',
                data={
                    'response_type': 'text',
                    'text': llm_response, 
                    'message': 'Analysis response'
                }
            )
            message = "Analysis completed"
        
        # Add to conversation history
        session.add_message('user', query)
        session.add_message('assistant', message, response_data.response_type, time.time() - start_time)
        
        execution_time = time.time() - start_time
        
        return response_data, message, execution_time
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        execution_time = time.time() - start_time
        
        error_response = QueryResponseData(
            response_type='error',
            data={
                'response_type': 'error',
                'error': str(e),
                'message': 'Query processing failed',
                'error_type': 'processing_error'
            }
        )
        
        session.add_message('user', query)
        session.add_message('assistant', f"Error: {str(e)}", 'error', execution_time)
        
        return error_response, f"Error processing query: {str(e)}", execution_time


def _get_data_context(df: pd.DataFrame) -> str:
    """Generate context about the dataset for LLM."""
    try:
        context_parts = []
        
        # Basic info
        context_parts.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        context_parts.append(f"Columns: {', '.join(df.columns)}")
        
        # Data types
        dtype_info = []
        for col, dtype in df.dtypes.items():
            dtype_info.append(f"{col} ({dtype})")
        context_parts.append(f"Data types: {', '.join(dtype_info)}")
        
        # Sample data
        sample = df.head(3).to_string(index=False)
        context_parts.append(f"Sample data:\n{sample}")
        
        return '\n\n'.join(context_parts)
        
    except Exception as e:
        logger.error(f"Error generating data context: {e}")
        return f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns"


def _generate_analysis_prompt(query: str, data_context: str, conversation_context: str) -> str:
    """Generate enhanced analysis prompt for LLM with intelligent visualization selection."""
    
    prompt = f"""
You are a world-class data scientist and visualization expert. Create data-driven insights with professional charts that tell compelling stories.

DATASET CONTEXT:
{data_context}

CONVERSATION HISTORY:
{conversation_context}

USER QUESTION: {query}

ANALYSIS METHODOLOGY:
🔍 UNDERSTAND: Analyze data structure, types, and distributions
📊 CHOOSE: Select optimal visualization based on data characteristics and query intent
✨ CREATE: Build professional, interactive visualizations with proper styling
💡 INTERPRET: Provide intelligent insights, trends, and actionable recommendations

AVAILABLE LIBRARIES:
• df: Your dataset (pandas DataFrame)
• pd, np: Data manipulation and numerical operations
• px: Plotly Express for elegant statistical visualizations
• go: Plotly Graph Objects for advanced custom charts
• scipy.stats: Statistical tests and distributions
• sklearn: Machine learning and preprocessing tools

VISUALIZATION DECISION TREE:

📈 NUMERICAL ANALYSIS:
   • Single variable: Histogram + box plot (px.histogram with marginal)
   • Two variables: Scatter plot with trendline (px.scatter with trendline)
   • Multiple variables: Correlation heatmap (px.imshow)
   • Distributions: Violin plots, box plots by category

📊 CATEGORICAL ANALYSIS:
   • Counts/frequencies: Horizontal bar charts (px.bar)
   • Proportions: Pie charts or treemap for hierarchical data
   • Category comparison: Grouped bar charts

📉 TIME SERIES:
   • Trends: Line charts with proper date formatting
   • Seasonal patterns: Multi-line plots by category
   • Comparisons: Area charts or multiple y-axes

🔗 RELATIONSHIPS:
   • Correlations: Heatmaps with annotations
   • Comparisons: Side-by-side plots or subplots
   • Statistical: Regression plots with confidence intervals

CODE TEMPLATE (ADAPT TO YOUR SPECIFIC ANALYSIS):
```python
# STEP 1: DATA EXPLORATION & UNDERSTANDING
print("=== DATASET ANALYSIS ===")
print(f"📊 Dataset: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print("\n📋 Column Details:")

for col in df.columns[:10]:  # Show first 10 columns
    dtype = df[col].dtype
    unique_count = df[col].nunique()
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df) * 100).round(1)
    
    if dtype in ['object', 'category']:
        top_values = df[col].value_counts().head(3).to_dict()
        print(f"  📝 {{col}}: {{dtype}} | {{unique_count}} unique | {{null_pct}}% null | Top: {{top_values}}")
    else:
        mean_val = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
        print(f"  🔢 {{col}}: {{dtype}} | {{unique_count}} unique | {{null_pct}}% null | Mean: {{mean_val:.2f}}" if mean_val != 'N/A' else f"  🔢 {{col}}: {{dtype}} | {{unique_count}} unique | {{null_pct}}% null")

# STEP 2: INTELLIGENT CHART SELECTION
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
date_cols = pd.to_datetime(df.select_dtypes(include=['object']).apply(pd.to_datetime, errors='coerce'), errors='coerce').dropna(axis=1).columns.tolist()

print(f"\n🎯 Available for analysis: {{len(numeric_cols)}} numeric, {{len(categorical_cols)}} categorical, {{len(date_cols)}} date columns")

# STEP 3: CREATE CONTEXT-AWARE VISUALIZATION
query_lower = query.lower()

# Enhanced chart selection logic with advanced analytics
if any(word in query_lower for word in ['correlation', 'relationship', 'association']) and len(numeric_cols) > 1:
    # Advanced correlation analysis with statistical significance
    corr_matrix = df[numeric_cols].corr()
    
    # Calculate p-values for correlations
    p_values = np.zeros((len(numeric_cols), len(numeric_cols)))
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i != j:
                corr_coef, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                p_values[i, j] = p_val
    
    # Create significance mask
    significance_mask = p_values < 0.05
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="🔗 Correlation Matrix with Statistical Significance",
                    color_continuous_scale='RdBu_r',
                    labels=dict(color="Correlation"))
    
    print("\n📊 Chart Type: Advanced Correlation Analysis with P-values")
    print(f"Strong correlations (|r| > 0.7): {((np.abs(corr_matrix) > 0.7) & (corr_matrix != 1)).sum().sum() // 2}")
    print(f"Significant correlations (p < 0.05): {significance_mask.sum() // 2}")
    
elif any(word in query_lower for word in ['trend', 'time', 'over time', 'temporal']) and len(date_cols) > 0:
    # Time series analysis
    target_col = numeric_cols[0] if numeric_cols else categorical_cols[0]
    fig = px.line(df, x=date_cols[0], y=target_col,
                  title=f"📈 Trend Analysis: {{target_col}} Over Time",
                  markers=True)
    fig.update_traces(line_width=3, marker_size=6)
    print(f"\n📊 Chart Type: Time Series Line Chart ({{target_col}} vs {{date_cols[0]}})")
    
elif any(word in query_lower for word in ['distribution', 'spread', 'histogram', 'outlier', 'anomaly']) and len(numeric_cols) > 0:
    # Advanced distribution analysis with outlier detection
    target_col = numeric_cols[0]
    
    # Detect outliers using IQR method
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[target_col] < (Q1 - 1.5 * IQR)) | (df[target_col] > (Q3 + 1.5 * IQR))]
    
    # Statistical tests for normality
    shapiro_stat, shapiro_p = stats.shapiro(df[target_col].dropna().sample(min(5000, len(df[target_col].dropna()))))
    
    fig = px.histogram(df, x=target_col, 
                       title=f"📊 Advanced Distribution Analysis: {{target_col}}",
                       marginal="box",
                       opacity=0.7,
                       nbins=30)
    
    # Add normal distribution curve overlay
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    x_range = np.linspace(df[target_col].min(), df[target_col].max(), 100)
    normal_curve = stats.norm.pdf(x_range, mean_val, std_val)
    
    fig.add_scatter(x=x_range, y=normal_curve * len(df) * (df[target_col].max() - df[target_col].min()) / 30,
                   mode='lines', name='Normal Distribution', line=dict(color='red', dash='dash'))
    
    print(f"\n📊 Chart Type: Advanced Distribution Analysis with Outlier Detection ({{target_col}})")
    print(f"Outliers detected: {{len(outliers)}} ({{{len(outliers)/len(df)*100:.1f}}}% of data)")
    print(f"Normality test (Shapiro-Wilk): p-value = {{shapiro_p:.4f}} ({'Normal' if shapiro_p > 0.05 else 'Not Normal'})")
    print(f"Skewness: {{stats.skew(df[target_col].dropna()):.3f}}")
    print(f"Kurtosis: {{stats.kurtosis(df[target_col].dropna()):.3f}}")
    
elif any(word in query_lower for word in ['compare', 'comparison', 'difference']) and len(categorical_cols) > 0 and len(numeric_cols) > 0:
    # Categorical comparison
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]
    summary_df = df.groupby(cat_col)[num_col].agg(['mean', 'count']).reset_index()
    summary_df.columns = [cat_col, 'Average', 'Count']
    
    fig = px.bar(summary_df, x=cat_col, y='Average',
                title=f"📊 Comparison: Average {{num_col}} by {{cat_col}}",
                text='Count',
                color='Average',
                color_continuous_scale='viridis')
    fig.update_traces(textposition='outside')
    print(f"\n📊 Chart Type: Categorical Comparison Bar Chart ({{num_col}} by {{cat_col}})")
    
elif len(numeric_cols) >= 2:
    # Scatter plot for relationship analysis
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    color_col = categorical_cols[0] if categorical_cols else None
    
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                    title=f"🔍 Relationship Analysis: {{x_col}} vs {{y_col}}",
                    trendline="ols",  # Add regression line
                    opacity=0.7,
                    size_max=10)
    print(f"\n📊 Chart Type: Scatter Plot with Regression Line ({{x_col}} vs {{y_col}})")
    
elif any(word in query_lower for word in ['cluster', 'segment', 'group', 'similar']) and len(numeric_cols) >= 2:
    # Clustering analysis
    cluster_data = df[numeric_cols].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Determine optimal number of clusters using elbow method
    max_k = min(10, len(cluster_data) // 10)
    if max_k >= 2:
        wcss = []
        k_range = range(2, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
        
        # Find optimal k using elbow method
        optimal_k = 3 if len(k_range) >= 3 else k_range[0]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data
        cluster_data['Cluster'] = clusters
        
        # Create scatter plot with clusters
        fig = px.scatter(cluster_data, x=numeric_cols[0], y=numeric_cols[1], 
                        color='Cluster', 
                        title=f"🔍 K-Means Clustering Analysis (k={{optimal_k}})",
                        color_discrete_sequence=px.colors.qualitative.Set1)
        
        print(f"\n📊 Chart Type: K-Means Clustering Analysis ({{optimal_k}} clusters)")
        print(f"Silhouette score: {{silhouette_score(scaled_data, clusters):.3f}}")
    else:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                        title="🔍 Data Scatter Plot",
                        opacity=0.7)
        print("\n📊 Chart Type: Basic Scatter Plot")
        
elif any(word in query_lower for word in ['pca', 'dimension', 'component', 'reduce']) and len(numeric_cols) >= 3:
    # Principal Component Analysis
    pca_data = df[numeric_cols].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
    
    fig = px.scatter(pca_df, x='PC1', y='PC2',
                    title=f"🔮 PCA Analysis: First 2 Components ({{pca.explained_variance_ratio_[:2].sum():.1%}} variance)",
                    opacity=0.7)
    
    print(f"\n📊 Chart Type: Principal Component Analysis")
    print(f"Explained variance by PC1: {{pca.explained_variance_ratio_[0]:.1%}}")
    print(f"Explained variance by PC2: {{pca.explained_variance_ratio_[1]:.1%}}")
    print(f"Cumulative explained variance: {{pca.explained_variance_ratio_[:2].sum():.1%}}")
    
elif any(word in query_lower for word in ['anomaly', 'outlier', 'unusual', 'detect']) and len(numeric_cols) >= 1:
    # Anomaly detection using Isolation Forest
    anomaly_data = df[numeric_cols].dropna()
    
    # Fit Isolation Forest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = isolation_forest.fit_predict(anomaly_data)
    
    # Add anomaly labels
    anomaly_data['Anomaly'] = ['Anomaly' if label == -1 else 'Normal' for label in anomaly_labels]
    
    if len(numeric_cols) >= 2:
        fig = px.scatter(anomaly_data, x=numeric_cols[0], y=numeric_cols[1],
                        color='Anomaly',
                        title=f"⚠️ Anomaly Detection: {{numeric_cols[0]}} vs {{numeric_cols[1]}}",
                        color_discrete_map={{'Normal': 'blue', 'Anomaly': 'red'}})
    else:
        fig = px.histogram(anomaly_data, x=numeric_cols[0], color='Anomaly',
                          title=f"⚠️ Anomaly Detection: {{numeric_cols[0]}} Distribution",
                          opacity=0.7)
    
    anomaly_count = (anomaly_labels == -1).sum()
    print(f"\n📊 Chart Type: Anomaly Detection using Isolation Forest")
    print(f"Anomalies detected: {{anomaly_count}} ({{anomaly_count/len(anomaly_data)*100:.1f}}% of data)")
    
else:
    # Enhanced default analysis with automatic insights
    if numeric_cols:
        target_col = numeric_cols[0]
        
        # Perform comprehensive statistical analysis
        desc_stats = df[target_col].describe()
        skewness = stats.skew(df[target_col].dropna())
        kurtosis = stats.kurtosis(df[target_col].dropna())
        
        fig = px.histogram(df, x=target_col,
                          title=f"📊 Comprehensive Analysis: {{target_col}} Distribution",
                          marginal="box",
                          opacity=0.7)
        
        # Add statistical annotations
        fig.add_vline(x=desc_stats['mean'], line_dash="dash", line_color="red", 
                     annotation_text="Mean")
        fig.add_vline(x=desc_stats['50%'], line_dash="dash", line_color="green", 
                     annotation_text="Median")
        
        print(f"\n📊 Chart Type: Enhanced Distribution Analysis ({{target_col}})")
        print(f"Mean: {{desc_stats['mean']:.3f}}, Median: {{desc_stats['50%']:.3f}}")
        print(f"Skewness: {{skewness:.3f}} ({'Right-skewed' if skewness > 0.5 else 'Left-skewed' if skewness < -0.5 else 'Approximately symmetric'})")
        print(f"Kurtosis: {{kurtosis:.3f}} ({'Heavy-tailed' if kurtosis > 0 else 'Light-tailed'})")
    else:
        target_col = categorical_cols[0] if categorical_cols else df.columns[0]
        value_counts = df[target_col].value_counts()
        
        # Calculate diversity metrics
        shannon_entropy = -sum((p := value_counts / len(df)) * np.log2(p))
        simpson_index = sum(p ** 2)
        
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    title=f"📊 Enhanced Categorical Analysis: {{target_col}} Frequency",
                    labels={{'x': target_col, 'y': 'Count'}})
        
        print(f"\n📊 Chart Type: Enhanced Categorical Analysis ({{target_col}})")
        print(f"Unique categories: {{len(value_counts)}}")
        print(f"Shannon entropy (diversity): {{shannon_entropy:.3f}}")
        print(f"Simpson index (dominance): {{simpson_index:.3f}}")

# STEP 4: PROFESSIONAL STYLING
fig.update_layout(
    title={{"text": fig.layout.title.text, "x": 0.5, "xanchor": "center"}},
    title_font=dict(size=18, family="Arial Black"),
    font=dict(size=12, family="Arial"),
    plot_bgcolor='rgba(248, 249, 250, 0.8)',
    paper_bgcolor='rgba(255, 255, 255, 1)',
    margin=dict(t=100, b=70, l=70, r=70),
    showlegend=True if 'color' in fig.data[0] else False,
    hovermode='closest'
)

# Add gridlines for better readability
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

fig.show()

# STEP 5: INTELLIGENT INSIGHTS & INTERPRETATION
print("\n=== 💡 KEY INSIGHTS & FINDINGS ===")
print("\n🔍 Data Characteristics:")
print(f"  • Dataset contains {{df.shape[0]:,}} observations across {{df.shape[1]}} variables")
print(f"  • Data quality: {{((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}}% complete")

if numeric_cols:
    print(f"\n📊 Numerical Analysis:")
    for col in numeric_cols[:3]:  # Top 3 numeric columns
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  • {{col}}: Mean = {{mean_val:.2f}}, Std = {{std_val:.2f}}")

if categorical_cols:
    print(f"\n📋 Categorical Analysis:")
    for col in categorical_cols[:3]:  # Top 3 categorical columns
        top_category = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
        unique_count = df[col].nunique()
        print(f"  • {{col}}: {{unique_count}} categories, most common = {{top_category}}")

print("\n🎯 Key Findings:")
print("  • [Main pattern or trend discovered in the visualization]")
print("  • [Statistical significance or notable relationships]")
print("  • [Outliers, anomalies, or interesting data points]")
print("  • [Data quality observations or missing value patterns]")

print("\n💼 Business Insights:")
print("  • [What this analysis means for decision-making]")
print("  • [Potential opportunities or risks identified]")
print("  • [Recommended actions based on findings]")

print("\n🔮 Next Steps:")
print("  • [Suggested follow-up analyses or questions]")
print("  • [Additional data that might be helpful]")
print("  • [Specific business metrics to monitor]")

print("\n" + "="*50)
print("📈 Analysis Complete - Chart Generated Successfully!")
print("="*50)
```

EXECUTION GUIDELINES:
✅ ALWAYS start with data exploration to understand structure
✅ Select the most appropriate chart type based on data characteristics
✅ Create professional, publication-ready visualizations
✅ Provide comprehensive insights that go beyond just describing the chart
✅ Include business implications and actionable recommendations
✅ Use proper statistical language and formatting
✅ Make every visualization tell a clear, compelling story

❌ AVOID generic charts that don't match the data
❌ Skip the insights section - it's mandatory
❌ Create misleading or unclear visualizations
❌ Use inappropriate chart types for the data structure
"""
    
    return prompt


def _extract_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from LLM response."""
    code_blocks = []
    lines = text.split('\n')
    in_code_block = False
    current_block = []
    
    for line in lines:
        if line.strip().startswith('```python') or line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                if current_block:
                    code_blocks.append('\n'.join(current_block))
                current_block = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
        elif in_code_block:
            current_block.append(line)
    
    # Handle case where code block doesn't end properly
    if current_block:
        code_blocks.append('\n'.join(current_block))
    
    return code_blocks


def _execute_code_safely(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute Python code safely with restricted imports and error handling.
    
    Args:
        code: Python code to execute
        df: DataFrame to work with
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Comprehensive security validation
        tree = ast.parse(code)
        
        # Dangerous functions and attributes to block
        dangerous_names = {
            '__import__', 'exec', 'eval', 'compile', 'open', 'file', 
            'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr', '__builtins__'
        }
        
        for node in ast.walk(tree):
            # Check imports - block ANY import statements since modules are pre-imported
            if isinstance(node, ast.Import):
                return {
                    'success': False,
                    'error': 'Import statements not allowed - all modules are pre-imported',
                    'output': ''
                }
            elif isinstance(node, ast.ImportFrom):
                return {
                    'success': False,
                    'error': 'Import statements not allowed - all modules are pre-imported',
                    'output': ''
                }
            # Check dangerous function calls
            elif isinstance(node, ast.Name) and node.id in dangerous_names:
                return {
                    'success': False,
                    'error': f'Use of "{node.id}" is not allowed for security reasons',
                    'output': ''
                }
            # Check attribute access to dangerous methods
            elif isinstance(node, ast.Attribute) and node.attr in dangerous_names:
                return {
                    'success': False,
                    'error': f'Access to "{node.attr}" is not allowed for security reasons',
                    'output': ''
                }
        
        # Create restricted execution environment
        safe_builtins = {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'max': max,
            'min': min,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'type': type,
            'isinstance': isinstance,
            'bool': bool,
            'any': any,
            'all': all,
            'map': map,
            'filter': filter,
            'reversed': reversed,
            'format': format,
            'repr': repr
            # Explicitly exclude dangerous functions like __import__, exec, eval, etc.
        }
        
        exec_globals = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'stats': stats,
            'StandardScaler': StandardScaler,
            'LabelEncoder': LabelEncoder,
            'PCA': PCA,
            'KMeans': KMeans,
            'IsolationForest': IsolationForest,
            'SelectKBest': SelectKBest,
            'f_regression': f_regression,
            'silhouette_score': silhouette_score,
            'df': df.copy()  # Work with a copy to avoid modifying original
        }
        
        exec_locals = {}
        
        # Capture output
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        try:
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, exec_globals, exec_locals)
                
            output = output_buffer.getvalue()
            error_output = error_buffer.getvalue()
            
            # Check for plotly figures in locals (get the last one created)
            figure_data = None
            fig_variable = None
            
            # Look for variables named 'fig' first, then any plotly figure
            if 'fig' in exec_locals and hasattr(exec_locals['fig'], 'to_json'):
                figure_data = json.loads(exec_locals['fig'].to_json())
                fig_variable = 'fig'
            else:
                # Look for any plotly figure object
                for key, value in exec_locals.items():
                    if hasattr(value, 'to_json') and hasattr(value, 'data'):  # More specific check for plotly figure
                        figure_data = json.loads(value.to_json())
                        fig_variable = key
                        break
            
            result = {
                'success': True,
                'output': output,
                'error': error_output,
                'locals': exec_locals,
                'figure': figure_data
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': output_buffer.getvalue()
            }
            
    except SyntaxError as e:
        return {
            'success': False,
            'error': f'Syntax error: {str(e)}',
            'output': ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output': ''
        }


def _process_execution_result(result: Dict[str, Any], query: str) -> QueryResponseData:
    """Process execution result into appropriate response format."""
    
    # Check if there's a plotly figure
    if result.get('figure'):
        return QueryResponseData(
            response_type='plot',
            data={
                'response_type': 'plot',
                'type': 'plotly',
                'data': result['figure'],
                'chart_type': _infer_chart_type(result['figure']),
                'title': _extract_title_from_figure(result['figure']) or 'Data Visualization'
            }
        )
    
    # Check if there are statistical results
    output = result.get('output', '')
    if any(keyword in output.lower() for keyword in ['mean', 'std', 'correlation', 'summary', 'describe']):
        stats_data = {
            'response_type': 'statistics',
            'calculation': 'Statistical Analysis',
            'results': result.get('locals', {}),
            'interpretation': output,
            'summary_stats': _extract_summary_stats(result.get('locals', {}))
        }
        return QueryResponseData(
            response_type='statistics',
            data=convert_numpy_types(stats_data)
        )
    
    # Check if this looks like an insight query
    if any(keyword in query.lower() for keyword in ['insight', 'trend', 'pattern', 'overview', 'summary']):
        return QueryResponseData(
            response_type='insight',
            data={
                'response_type': 'insight',
                'summary': 'Data Analysis Complete',
                'key_insights': _extract_insights_from_output(output),
                'data_quality': {'status': 'analyzed'},
                'suggested_queries': _generate_suggested_queries(query)
            }
        )
    
    # Default to text response
    return QueryResponseData(
        response_type='text',
        data={
            'response_type': 'text',
            'text': output,
            'message': 'Analysis completed successfully'
        }
    )


def _infer_chart_type(figure_data: Dict[str, Any]) -> str:
    """Infer chart type from plotly figure data."""
    try:
        if 'data' in figure_data and figure_data['data']:
            trace_type = figure_data['data'][0].get('type', 'scatter')
            return trace_type
        return 'unknown'
    except:
        return 'unknown'


def _extract_title_from_figure(figure_data: Dict[str, Any]) -> Optional[str]:
    """Extract title from plotly figure data."""
    try:
        layout = figure_data.get('layout', {})
        return layout.get('title', {}).get('text')
    except:
        return None


def _extract_summary_stats(locals_dict: Dict[str, Any]) -> Dict[str, float]:
    """Extract summary statistics from execution locals."""
    stats = {}
    for key, value in locals_dict.items():
        if isinstance(value, (int, float)):
            stats[key] = float(value)
        elif isinstance(value, pd.Series) and value.dtype in ['int64', 'float64']:
            stats[f"{key}_mean"] = float(value.mean())
            stats[f"{key}_std"] = float(value.std())
    
    return convert_numpy_types(stats)


def _extract_insights_from_output(output: str) -> List[str]:
    """Extract key insights from execution output."""
    lines = output.strip().split('\n')
    insights = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('   ', '\t')):  # Skip indented lines (likely data output)
            insights.append(line)
    
    return insights[:5]  # Limit to 5 key insights


def _generate_suggested_queries(current_query: str) -> List[str]:
    """Generate suggested follow-up queries based on current query."""
    base_suggestions = [
        "Show me a visualization of this data",
        "What are the statistical summaries?",
        "Are there any missing values?",
        "What are the data types of each column?",
        "Show me correlation analysis"
    ]
    
    # Filter out suggestions similar to current query
    filtered = []
    for suggestion in base_suggestions:
        if not any(word in current_query.lower() for word in suggestion.lower().split()):
            filtered.append(suggestion)
    
    return filtered[:3]