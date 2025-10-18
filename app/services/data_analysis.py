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
        return float(obj)
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
5. CRITICAL: DO NOT include ANY import statements - all modules are already imported (pd, np, px, go)
6. CRITICAL: DO NOT use __import__, exec, eval, or any dynamic code execution
7. Available modules: pd (pandas), np (numpy), px (plotly.express), go (plotly.graph_objects)
8. For summaries: use clear section headers like "### Dataset Overview ###"
9. For visualizations: create ONE figure per request and use fig.show() at the end
10. IMPORTANT: Do NOT use customdata or hovertemplate in Plotly charts - use simple hover_data parameter only
11. For simple queries (overview, describe, summary), just use print() - NO visualization needed

FORMATTING GUIDELINES:
- Use "###" for main section headers
- Use "---" for separators
- Use bullet points for lists
- Add blank lines between sections
- Format numbers with appropriate precision

VISUALIZATION GUIDELINES:
- For pie charts: Use px.pie(df, names='column', values='column', title='Title')
- For bar charts: Use px.bar(df, x='column', y='column', title='Title')
- For scatter plots: Use px.scatter(df, x='column', y='column', title='Title')
- DO NOT manually set hovertemplate or customdata
- Let Plotly handle hover data automatically
- After creating visualizations, print a clear analysis of what the chart shows

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

Focus on:
- Clear insights and findings after EVERY visualization
- What patterns, trends, or anomalies are visible
- Data quality issues if any (outliers, missing values, imbalanced data)
- Actionable recommendations for the user
- Statistical analysis when relevant

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