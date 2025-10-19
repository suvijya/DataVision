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
from itertools import combinations, permutations

import google.generativeai as genai

# Import statistical libraries
try:
    import scipy
    import scipy.stats as stats
    from scipy import stats as sp_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    sp_stats = None
    
try:
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None
    adfuller = None
    grangercausalitytests = None

try:
    import sklearn
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

from app.core.simple_config import settings
from app.services.session_manager import Session

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel(settings.LLM_MODEL)


def convert_numpy_types(obj):
    """Convert NumPy and Pandas types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN and Infinity
        val = float(obj)
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return None  # or use a large number like 1e308
        return val
    elif isinstance(obj, (float, int)):
        # Handle regular Python floats that might be NaN/Inf
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to a dict representation
        try:
            return {
                'type': 'DataFrame',
                'shape': obj.shape,
                'columns': obj.columns.tolist(),
                'sample_data': obj.head(5).to_dict('records')
            }
        except:
            return {'type': 'DataFrame', 'shape': obj.shape, 'columns': obj.columns.tolist()}
    elif isinstance(obj, pd.Series):
        # Convert Series to a dict representation
        try:
            return {
                'type': 'Series',
                'name': obj.name,
                'length': len(obj),
                'dtype': str(obj.dtype),
                'sample_values': obj.head(5).tolist()
            }
        except:
            return {'type': 'Series', 'name': obj.name, 'length': len(obj)}
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        # Handle other pandas objects with to_dict method
        try:
            return convert_numpy_types(obj.to_dict())
        except:
            return str(obj)
    else:
        # Filter out non-serializable objects (sklearn models, complex objects, etc.)
        # Check if it's a simple type that can be JSON serialized
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        # Try to check if it's a sklearn model or other complex object
        obj_type = type(obj).__name__
        obj_module = type(obj).__module__
        
        # Skip sklearn models, statsmodels, and other ML objects
        if any(module in obj_module for module in ['sklearn', 'statsmodels', 'scipy.stats._', 'plotly']):
            return f"<{obj_type} object - not serializable>"
        
        # Try JSON serialization test
        try:
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If it can't be serialized, return a string representation
            return f"<{obj_type} object>"


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
    """Generate analysis prompt for LLM."""
    
    prompt = f"""
You are a data analysis assistant. Analyze the provided dataset and answer the user's query.

DATASET CONTEXT:
{data_context}

CONVERSATION HISTORY:
{conversation_context}

USER QUERY: {query}

INSTRUCTIONS:
1. Write Python code to analyze the data and answer the query
2. Use pandas for data manipulation (df is the DataFrame variable)
3. For visualizations, use plotly.express (px) or plotly.graph_objects (go)
4. Format output text with clear headers, bullet points, and proper spacing
5. ⚠️ CRITICAL: DO NOT include ANY import statements - all modules are ALREADY IMPORTED
6. ⚠️ CRITICAL: DO NOT use __import__, exec, eval, or any dynamic code execution
7. ⚠️ CRITICAL: DO NOT return model objects (LinearRegression, IsolationForest, etc.) - only return RESULTS (numbers, strings, arrays)
8. ⚠️ CRITICAL: For statistical analysis - ONLY create visualizations if query EXPLICITLY says "visualize", "create chart", "plot", or "show graph"
9. ⚠️ CRITICAL: If query says "TEXT FORMAT ONLY" or "Analyze" - use ONLY print() statements, NO visualizations
10. ⚠️ DO NOT write: from sklearn import..., import scipy, import statsmodels, etc.
11. Available modules (PRE-IMPORTED - USE DIRECTLY):
   - pd (pandas), np (numpy), px (plotly.express), go (plotly.graph_objects)
   - stats, sp_stats (scipy.stats) - for statistical tests and p-value calculations
   - sm (statsmodels.api) - ONLY for advanced models (ARIMA, VAR, etc.) - DO NOT use for simple regression
   - adfuller, grangercausalitytests (statsmodels) - for time series
   - LinearRegression, LogisticRegression, PolynomialFeatures (sklearn - USE DIRECTLY)
   - IsolationForest (sklearn.ensemble) - for outlier detection
   - mean_squared_error, r2_score (sklearn.metrics)
   - train_test_split (sklearn.model_selection)
   - combinations, permutations (itertools) - for variable pair analysis
12. For summaries: use clear section headers like "### Dataset Overview ###"
13. For visualizations: Create charts ONLY when explicitly requested in the query
14. For simple queries (overview, describe, summary), just use print() - NO visualization needed
15. ⚠️ CRITICAL: For LINEAR REGRESSION - use sklearn LinearRegression, NOT statsmodels sm.OLS
16. ⚠️ For p-values in regression - calculate manually using scipy.stats (already imported as 'stats')

STATISTICAL ANALYSIS EXAMPLES (NO IMPORTS NEEDED):
- Normality Test: stats.shapiro(df['column'])
- T-Test: stats.ttest_ind(df[df['group']=='A']['value'], df[df['group']=='B']['value'])
- Correlation: df[['col1', 'col2']].corr(method='pearson')
- ANOVA: stats.f_oneway(group1, group2, group3)
- Chi-Square: stats.chi2_contingency(pd.crosstab(df['cat1'], df['cat2']))
- Variable Pairs: list(combinations(numeric_cols, 2))  # combinations is pre-imported
- Outliers (IQR): Q1 = df['col'].quantile(0.25); Q3 = df['col'].quantile(0.75); IQR = Q3 - Q1
- Outliers (Z-score): np.abs(stats.zscore(df['col'])) > 3
- Outliers (Isolation Forest): IsolationForest(contamination=0.1).fit_predict(df[['col']])
- Linear Regression (TEXT OUTPUT ONLY - DO NOT VISUALIZE):
  # Use sklearn LinearRegression for simple regression (DO NOT use sm/statsmodels for basic regression)
  # Remove rows with NaN values in predictor or target columns
  valid_data = df[['predictor_col', 'target_col']].dropna()
  X = valid_data[['predictor_col']].values
  y = valid_data['target_col'].values
  n_removed = len(df) - len(valid_data)
  model = LinearRegression()
  model.fit(X, y)
  predictions = model.predict(X)
  r2 = r2_score(y, predictions)
  rmse = np.sqrt(mean_squared_error(y, predictions))
  
  # Calculate p-values using scipy.stats (already imported as 'stats')
  n = len(valid_data)
  residuals = y - predictions
  residual_std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
  x_mean = np.mean(X)
  x_std = np.sum((X - x_mean)**2)
  se_coef = residual_std_error / np.sqrt(x_std)
  t_stat = model.coef_[0] / se_coef
  p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
  
  print(f"### Linear Regression Results ###")
  print(f"Predictor: predictor_col → Target: target_col")
  if n_removed > 0:
      print(f"ℹ️ Removed {{n_removed}} rows with missing values ({{n_removed/len(df)*100:.1f}}% of data)")
  print(f"Sample size: {{len(valid_data)}} rows")
  print(f"R² Score: {{r2:.4f}}")
  print(f"RMSE: {{rmse:.4f}}")
  print(f"Coefficient: {{model.coef_[0]:.4f}} (p-value: {{p_value:.4f}})")
  print(f"Intercept: {{model.intercept_:.4f}}")
  print(f"\\n### Interpretation ###")
  if r2 < 0.1:
      print(f"⚠️ VERY WEAK: R²={{r2:.4f}} means these variables have almost NO linear relationship.")
      print(f"💡 Suggestion: Try different variable pairs. Exclude ID columns, postal codes, or indices.")
  elif r2 < 0.3:
      print(f"⚠️ WEAK: R²={{r2:.4f}} means only {{r2*100:.1f}}% of variance is explained.")
      print(f"💡 Suggestion: This model has limited predictive power. Try different variables.")
  elif r2 < 0.7:
      print(f"✓ MODERATE: R²={{r2:.4f}} means {{r2*100:.1f}}% of variance is explained.")
  else:
      print(f"✅ STRONG: R²={{r2:.4f}} means {{r2*100:.1f}}% of variance is explained.")
  print(f"\\nEquation: target_col = {{model.coef_[0]:.4f}} * predictor_col + {{model.intercept_:.4f}}")
  print(f"\\nFirst 10 predictions:")
  for i in range(min(10, len(predictions))):
      print(f"  Actual: {{y[i]:.2f}}, Predicted: {{predictions[i]:.2f}}, Residual: {{y[i]-predictions[i]:.2f}}")
  # DO NOT ADD fig.show() or any visualization!

- Polynomial Regression (TEXT OUTPUT ONLY - DO NOT VISUALIZE):
  # Remove rows with NaN values
  valid_data = df[['predictor_col', 'target_col']].dropna()
  X = valid_data[['predictor_col']].values
  y = valid_data['target_col'].values
  poly = PolynomialFeatures(degree=2)
  X_poly = poly.fit_transform(X)
  model = LinearRegression()
  model.fit(X_poly, y)
  predictions = model.predict(X_poly)
  r2 = r2_score(y, predictions)
  rmse = np.sqrt(mean_squared_error(y, predictions))
  print(f"### Polynomial Regression Results (Degree 2) ###")
  print(f"R² Score: {{r2:.4f}}")
  print(f"RMSE: {{rmse:.4f}}")
  # DO NOT ADD fig.show() or any visualization!

- Find Best Variable Pair for Regression:
  # Exclude ID-like columns
  exclude_keywords = ['id', 'postal', 'code', 'index', 'row']
  numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                  if not any(kw in col.lower() for kw in exclude_keywords)]
  # Calculate all pairwise correlations (combinations is pre-imported)
  pairs = []
  for col1, col2 in combinations(numeric_cols, 2):
      corr = df[[col1, col2]].corr().iloc[0, 1]
      pairs.append((abs(corr), corr, col1, col2))
  pairs.sort(reverse=True)
  print(f"### Top 3 Variable Pairs for Linear Regression ###")
  for i, (abs_corr, corr, c1, c2) in enumerate(pairs[:3], 1):
      print(f"{{i}}. {{c1}} → {{c2}}: r={{corr:.4f}}, R²≈{{corr**2:.4f}}")

- Distribution Fitting: stats.norm.fit(df['col']); stats.kstest(df['col'], 'norm', params)

VISUALIZATION EXAMPLES (ONLY when query explicitly requests visualization):

- Linear Regression Visualization (ONLY if query says "visualize" or "create chart"):
  # Remove rows with NaN values
  valid_data = df[['predictor_col', 'target_col']].dropna()
  X = valid_data[['predictor_col']].values
  y = valid_data['target_col'].values
  model = LinearRegression()
  model.fit(X, y)
  predictions = model.predict(X)
  r2 = r2_score(y, predictions)
  # Create simple DataFrame for Plotly
  viz_df = pd.DataFrame({{'x': valid_data['predictor_col'], 'Actual': y, 'Predicted': predictions}})
  fig = px.scatter(viz_df, x='x', y='Actual', title=f'Linear Regression (R²={{r2:.3f}})')
  fig.add_scatter(x=viz_df['x'], y=viz_df['Predicted'], mode='lines', name='Fit')
  if r2 < 0.3:
      fig.add_annotation(text=f"⚠️ Weak Correlation (R²={{r2:.3f}})", 
                         xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False,
                         font=dict(color="red", size=14))
  fig.show()

- Box Plot for Outliers (ONLY if query says "visualize"):
  fig = px.box(df, y='column', title='Outlier Detection')
  fig.show()

- Histogram with Normal Curve (ONLY if query says "visualize"):
  fig = px.histogram(df, x='column', nbins=30, title='Normality Test')
  fig.show()

- Correlation Scatter Plot (ONLY if query says "visualize"):
  corr = df[['col1', 'col2']].corr().iloc[0,1]
  fig = px.scatter(df, x='col1', y='col2', title=f'Correlation={{corr:.3f}}', trendline='ols')
  fig.show()

FORMATTING GUIDELINES:
- Use "###" for main section headers
- Use "---" for separators
- Use bullet points for lists
- Add blank lines between sections
- Format numbers with appropriate precision

VISUALIZATION GUIDELINES - 25+ Chart Types Available:

**Basic Charts:**
- Pie Chart: px.pie(df, names='category_col', values='value_col', title='Title')
- Bar Chart: px.bar(df, x='column', y='column', title='Title', color='category_col')
- Histogram: px.histogram(df, x='column', nbins=30, title='Distribution of X')
- Line Chart: px.line(df, x='time_col', y='value_col', title='Trend over Time')

**Distribution & Comparison:**
- Box Plot: px.box(df, x='category', y='numeric', title='Distribution by Category')
- Violin Plot: px.violin(df, x='category', y='numeric', box=True, title='Distribution Comparison')
- Strip Plot: px.strip(df, x='category', y='numeric', title='Individual Points by Category')
- Density Heatmap: px.density_heatmap(df, x='col1', y='col2', title='2D Distribution')

**Relationships:**
- Scatter Plot: px.scatter(df, x='col1', y='col2', color='category', size='size_col', title='Relationship')
- Bubble Chart: px.scatter(df, x='col1', y='col2', size='col3', color='col4', title='Multi-dimensional View')
- 3D Scatter: px.scatter_3d(df, x='col1', y='col2', z='col3', color='category', title='3D Visualization')
- Scatter Matrix: px.scatter_matrix(df, dimensions=['col1', 'col2', 'col3'], title='Pairwise Relationships')

**Correlations:**
- Heatmap (use go.Heatmap): fig = go.Figure(data=go.Heatmap(z=df.corr(), x=df.columns, y=df.columns, colorscale='RdBu'))
- Correlation Matrix with px: Create correlation first, then use px.imshow(df.corr(), title='Correlations')

**Hierarchical & Parts:**
- Sunburst: px.sunburst(df, path=['level1', 'level2'], values='values', title='Hierarchical View')
- Treemap: px.treemap(df, path=['category', 'subcategory'], values='values', title='Nested Rectangles')
- Icicle: px.icicle(df, path=['level1', 'level2'], values='values', title='Vertical Hierarchy')

**Trends & Time Series:**
- Area Chart: px.area(df, x='date', y='value', title='Cumulative Trend')
- Multiple Lines: px.line(df, x='date', y=['series1', 'series2'], title='Compare Trends')
- Range Area: Use go.Scatter with fill for confidence intervals

**Specialized Charts:**
- Funnel: px.funnel(df, x='values', y='stages', title='Conversion Funnel')
- Funnel Area: px.funnel_area(names=df['stage'], values=df['count'], title='Funnel Stages')
- Parallel Coordinates: px.parallel_coordinates(df, dimensions=['col1', 'col2', 'col3'], title='Multivariate')
- Parallel Categories: px.parallel_categories(df, dimensions=['cat1', 'cat2', 'cat3'], title='Flow Diagram')

**Financial & Special:**
- Candlestick (use go): fig = go.Figure(data=go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close']))
- Waterfall (use go): fig = go.Figure(go.Waterfall(x=df['categories'], y=df['values']))
- Indicator: fig = go.Figure(go.Indicator(mode='gauge+number', value=value, title='KPI'))

**IMPORTANT RULES:**
- DO NOT manually set hovertemplate or customdata
- Let Plotly handle hover data automatically using hover_data parameter
- After creating visualizations, print a clear analysis of what the chart shows
- Choose the chart type that best represents the data relationship

RESPONSE FORMAT:
Provide your analysis and then include the Python code in markdown code blocks like this:

**For overview/summary queries (NO visualization):**
```python
# NO IMPORTS - Just use print() and pandas methods
print("### Dataset Overview ###")
print("")
print(f"Shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print("")
print("Columns:", ', '.join(df.columns))
print("")
print("### Data Types ###")
print(df.dtypes.to_string())
print("")
print("### Summary Statistics ###")
print(df.describe().to_string())
```

**For visualization queries:**
```python
# NO IMPORTS - modules already available: pd, np, px, go, df
# Create ONE chart
fig = px.histogram(df, x='column_name', title='Clear Title')
fig.show()  # This MUST be the last line for visualizations

# IMPORTANT: After fig.show(), print insights about the visualization
print("")
print("### 📊 Analysis Insights ###")
print("")
print("**What this chart reveals:**")
print("• Key finding 1 from the visualization")
print("• Key finding 2 about the data distribution")
print("• Notable patterns or trends observed")
print("")
print("**Recommendations:**")
print("• Suggestion 1 for data quality or next steps")
print("• Suggestion 2 for further analysis if needed")
```

ADVANCED STATISTICAL ANALYSIS - Available via scipy.stats:
You now have access to advanced statistical tests via scipy.stats module (imported as stats):

**Normality Tests:**
```python
from scipy import stats
# Shapiro-Wilk test
stat, p = stats.shapiro(df['column'].dropna())
print(f"Shapiro-Wilk: statistic={{stat:.4f}}, p-value={{p:.4f}}")
print(f"Data is {{'normal' if p > 0.05 else 'NOT normal'}} (α=0.05)")
```

**T-Tests:**
```python
from scipy import stats
# Independent samples t-test
group1 = df[df['group']=='A']['value'].dropna()
group2 = df[df['group']=='B']['value'].dropna()
stat, p = stats.ttest_ind(group1, group2)
print(f"T-test: t={{stat:.4f}}, p={{p:.4f}}, significant={{p < 0.05}}")
```

**Correlation Tests:**
```python
from scipy import stats
# Pearson correlation
corr, p = stats.pearsonr(df['x'].dropna(), df['y'].dropna())
print(f"Pearson r={{corr:.4f}}, p={{p:.4f}}, significant={{p < 0.05}}")
```

**Chi-Square Test:**
```python
from scipy import stats
contingency = pd.crosstab(df['cat1'], df['cat2'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square={{chi2:.4f}}, p={{p:.4f}}, df={{dof}}")
```

**ANOVA:**
```python
from scipy import stats
groups = [group['value'].dropna() for name, group in df.groupby('category')]
f_stat, p = stats.f_oneway(*groups)
print(f"ANOVA: F={{f_stat:.4f}}, p={{p:.4f}}")
```

**Outlier Detection:**
```python
# Z-score method
z_scores = np.abs(stats.zscore(df['column'].dropna()))
outliers = df['column'][z_scores > 3]
print(f"Found {{len(outliers)}} outliers using Z-score method")

# IQR method
Q1, Q3 = df['column'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers_iqr = df['column'][(df['column'] < Q1 - 1.5*IQR) | (df['column'] > Q3 + 1.5*IQR)]
print(f"Found {{len(outliers_iqr)}} outliers using IQR method")
```

**Regression Analysis:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[['feature1', 'feature2']].dropna()
y = df.loc[X.index, 'target']

model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))

print(f"Linear Regression R²={{r2:.4f}}")
print(f"Coefficients: {{model.coef_}}")
print(f"Intercept: {{model.intercept_:.4f}}")
```

IMPORTANT: For statistical queries, always import the necessary modules at the top:
```python
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
```

Focus on:
- Clear insights and findings after EVERY visualization
- What patterns, trends, or anomalies are visible
- Data quality issues if any (outliers, missing values, imbalanced data)
- Actionable recommendations for the user
- Statistical analysis when relevant - use scipy.stats for hypothesis tests
- For advanced statistics, suggest using the dedicated /api/v1/statistical-analysis endpoints

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
            'df': df.copy(),  # Work with a copy to avoid modifying original
            'combinations': combinations,
            'permutations': permutations
        }
        
        # Add statistical libraries if available
        if SCIPY_AVAILABLE:
            exec_globals['scipy'] = scipy
            exec_globals['stats'] = stats
            exec_globals['sp_stats'] = sp_stats
            
        if STATSMODELS_AVAILABLE:
            exec_globals['statsmodels'] = statsmodels
            exec_globals['sm'] = sm
            exec_globals['adfuller'] = adfuller
            exec_globals['grangercausalitytests'] = grangercausalitytests
            
        if SKLEARN_AVAILABLE:
            exec_globals['sklearn'] = sklearn
            exec_globals['IsolationForest'] = IsolationForest
            exec_globals['LinearRegression'] = LinearRegression
            exec_globals['LogisticRegression'] = LogisticRegression
            exec_globals['PolynomialFeatures'] = PolynomialFeatures
            exec_globals['mean_squared_error'] = mean_squared_error
            exec_globals['r2_score'] = r2_score
            exec_globals['train_test_split'] = train_test_split
        
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
    
    # Filter out non-serializable objects from locals (sklearn models, etc.)
    def filter_serializable_locals(locals_dict):
        """Remove non-serializable objects like sklearn models from locals."""
        filtered = {}
        for key, value in locals_dict.items():
            # Skip private variables
            if key.startswith('_'):
                continue
            
            # Skip known non-serializable types
            value_type = type(value).__name__
            value_module = type(value).__module__
            
            # Skip sklearn models and other ML objects
            if any(module in value_module for module in ['sklearn', 'statsmodels.', 'scipy.stats._']):
                continue
            
            # Skip functions and classes
            if callable(value) and not isinstance(value, type):
                continue
            
            # Keep only serializable data
            filtered[key] = value
        
        return filtered
    
    # Check if there's a plotly figure
    if result.get('figure'):
        # Include the output (print statements) as insights for the visualization
        output_text = result.get('output', '').strip()
        
        logger.info(f"Processing plot result - output_text length: {len(output_text)}")
        logger.info(f"Output text preview: {output_text[:200] if output_text else 'EMPTY'}")
        
        plot_data = {
            'response_type': 'plot',
            'type': 'plotly',
            'data': result['figure'],
            'chart_type': _infer_chart_type(result['figure']),
            'title': _extract_title_from_figure(result['figure']) or 'Data Visualization'
        }
        
        # Add insights if there's meaningful output (not just the figure JSON)
        if output_text and not output_text.startswith('{'):
            plot_data['insights'] = output_text
            logger.info(f"Added insights to plot_data: {len(output_text)} chars")
        else:
            logger.warning(f"No insights added - output_text empty or starts with '{{': {output_text[:50] if output_text else 'EMPTY'}")
        
        return QueryResponseData(
            response_type='plot',
            data=plot_data
        )
    
    # Filter locals to remove non-serializable objects
    filtered_locals = filter_serializable_locals(result.get('locals', {}))
    
    # Check if there are statistical results
    output = result.get('output', '')
    if any(keyword in output.lower() for keyword in ['mean', 'std', 'correlation', 'summary', 'describe']):
        stats_data = {
            'response_type': 'statistics',
            'calculation': 'Statistical Analysis',
            'results': filtered_locals,
            'interpretation': output,
            'summary_stats': _extract_summary_stats(filtered_locals)
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
        elif isinstance(value, pd.Series) and pd.api.types.is_numeric_dtype(value):
            try:
                stats[f"{key}_mean"] = float(value.mean())
                stats[f"{key}_std"] = float(value.std())
            except:
                pass
        elif isinstance(value, pd.DataFrame):
            # Convert DataFrame to summary statistics
            try:
                numeric_cols = value.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    stats[f"{key}_{col}_mean"] = float(value[col].mean())
                    stats[f"{key}_{col}_std"] = float(value[col].std())
            except:
                stats[f"{key}_shape"] = f"{value.shape[0]}x{value.shape[1]}"
    
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