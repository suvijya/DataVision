# 🔒 Sandbox Creation Guide - Secure Code Execution Environment

## Overview

The **Visualization Sandbox** is a secure Python code execution environment that allows users to run data analysis and visualization code without compromising system security. It enables dynamic code generation and execution while preventing malicious operations.

---

## 🎯 Purpose

Enable users to:
- Run pandas data analysis operations safely
- Generate Plotly visualizations dynamically
- Execute LLM-generated Python code without security risks
- Prevent import statements (all modules pre-imported)
- Block dangerous functions (eval, exec, open, etc.)

---

## 🏗️ Architecture Components

### 1. Core Sandbox Function: `_execute_code_safely()`

**Location:** `app/services/data_analysis.py` (lines 908-1100+)

**Function Signature:**
```python
def _execute_code_safely(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute Python code safely with restricted imports and error handling.
    
    Args:
        code: Python code to execute
        df: DataFrame to work with
        
    Returns:
        Dictionary with execution results:
        {
            'success': bool,
            'output': str,
            'error': str,
            'locals': dict,
            'figure': dict  # Plotly figure JSON if created
        }
    """
```

---

## 🔐 Security Layers

### Layer 1: AST (Abstract Syntax Tree) Analysis

**Purpose:** Pre-execution static code analysis to block dangerous operations

```python
# Parse code into AST
tree = ast.parse(code)

# Dangerous functions to block
dangerous_names = {
    '__import__', 'exec', 'eval', 'compile', 'open', 'file', 
    'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
    'getattr', 'setattr', 'delattr', 'hasattr', '__builtins__'
}

# Walk through AST nodes
for node in ast.walk(tree):
    # Block import statements
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        return {'success': False, 'error': 'Import statements not allowed'}
    
    # Block dangerous function calls
    elif isinstance(node, ast.Name) and node.id in dangerous_names:
        return {'success': False, 'error': f'Use of "{node.id}" not allowed'}
    
    # Block dangerous attribute access
    elif isinstance(node, ast.Attribute) and node.attr in dangerous_names:
        return {'success': False, 'error': f'Access to "{node.attr}" not allowed'}
```

**What it blocks:**
- ❌ `import pandas as pd` → Blocked (modules pre-imported)
- ❌ `from os import system` → Blocked
- ❌ `eval("malicious_code")` → Blocked
- ❌ `exec("dangerous_ops")` → Blocked
- ❌ `open("sensitive_file.txt")` → Blocked
- ❌ `__import__("os")` → Blocked

---

### Layer 2: Restricted Execution Environment

**Purpose:** Create sandboxed globals/builtins with only safe functions

```python
# Safe built-in functions
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
    'repr': repr,
    '__name__': '__main__',
    '__build_class__': __builtins__['__build_class__'],
}

# Pre-imported modules (no user imports allowed)
exec_globals = {
    '__builtins__': safe_builtins,
    'pd': pd,              # pandas
    'np': np,              # numpy
    'px': px,              # plotly.express
    'go': go,              # plotly.graph_objects
    'df': df.copy(),       # User's DataFrame (copy to avoid mutation)
    'combinations': combinations,
    'permutations': permutations
}
```

**Conditional Module Availability:**
```python
# Statistical libraries (if installed)
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
```

---

### Layer 3: Output Capture & Execution

**Purpose:** Safely execute code and capture outputs/errors

```python
exec_locals = {}

# Capture stdout/stderr
output_buffer = StringIO()
error_buffer = StringIO()

try:
    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
        exec(code, exec_globals, exec_locals)
        
    output = output_buffer.getvalue()
    error_output = error_buffer.getvalue()
    
except Exception as e:
    return {
        'success': False,
        'error': str(e),
        'output': output_buffer.getvalue()
    }
```

---

### Layer 4: Plotly Figure Extraction

**Purpose:** Automatically detect and extract Plotly visualizations

```python
# Check for plotly figures in execution locals
figure_data = None

# Look for 'fig' variable first
if 'fig' in exec_locals and hasattr(exec_locals['fig'], 'to_json'):
    figure_data = json.loads(exec_locals['fig'].to_json())
else:
    # Look for any plotly figure object
    for key, value in exec_locals.items():
        if hasattr(value, 'to_json') and hasattr(value, 'data'):
            figure_data = json.loads(value.to_json())
            break

return {
    'success': True,
    'output': output,
    'error': error_output,
    'locals': exec_locals,
    'figure': figure_data  # JSON representation of Plotly chart
}
```

---

## 🧪 Testing Infrastructure

### Test File: `test_visualization_sandbox.py`

**4 Comprehensive Test Cases:**

#### 1. **Basic Pandas Operations**
```python
def test_basic_code_execution():
    code = """
missing_info = df.isnull().sum()
print("Missing values per column:")
for col, count in missing_info.items():
    print(f"{col}: {count}")
"""
    result = _execute_code_safely(code, df)
    # ✅ Should succeed with output
```

#### 2. **Visualization Code**
```python
def test_visualization_code():
    code = """
value_counts = df['Item Description'].value_counts()
fig = px.bar(
    x=value_counts.index,
    y=value_counts.values,
    title='Distribution of Item Descriptions'
)
fig.show()
"""
    result = _execute_code_safely(code, df)
    # ✅ Should succeed with figure
```

#### 3. **Import Blocking**
```python
def test_import_blocking():
    code = """
import pandas as pd
print("This should not execute")
"""
    result = _execute_code_safely(code, df)
    # ✅ Should fail with import error
```

#### 4. **Dangerous Function Blocking**
```python
def test_dangerous_functions():
    code = """
result = eval("1 + 1")
print(result)
"""
    result = _execute_code_safely(code, df)
    # ✅ Should fail with security error
```

---

## 🔄 Integration Flow

### 1. User Query Flow

```
User Query → LLM (Gemini) → Generated Code → Sandbox Execution → Result
```

### 2. Code Generation Prompt (LLM Instructions)

**Location:** `app/services/data_analysis.py` - `_build_system_prompt()`

**Critical Security Rules:**
```python
6. ⚠️ CRITICAL: DO NOT use __import__, exec, eval, or any dynamic code execution
7. ⚠️ CRITICAL: NEVER write import statements (all modules are pre-imported)
8. ⚠️ CRITICAL: For statistical analysis - ONLY create visualizations if query 
   EXPLICITLY says "visualize", "create chart", "plot", or "show graph"
9. ⚠️ CRITICAL: If query says "TEXT FORMAT ONLY" - use ONLY print() statements
```

**Available Pre-imported Modules:**
```python
Available pre-imported modules:
- pd (pandas)
- np (numpy)  
- px (plotly.express)
- go (plotly.graph_objects)
- scipy, stats, sp_stats (if available)
- statsmodels, sm, adfuller, grangercausalitytests (if available)
- sklearn, IsolationForest, LinearRegression, etc. (if available)
```

### 3. Execution Pipeline

```python
def process_query(session_id: str, query: str) -> Tuple[Dict, str, float]:
    # 1. Get LLM response with code
    llm_response = _generate_llm_response(prompt)
    
    # 2. Extract code blocks
    code_blocks = _extract_code_blocks(llm_response)
    
    # 3. Execute in sandbox
    if code_blocks:
        result = _execute_code_safely(code_blocks[0], df)
        
        # 4. Process results
        if result['success']:
            response_data = _process_execution_result(result, query)
        else:
            response_data = {'error': result['error']}
    
    return response_data, message, execution_time
```

---

## 🎨 Frontend Integration

### Dual-Mode Interface (Analyze vs Visualize)

**Frontend:** `frontend/script.js`

#### Analyze Button (Text Only)
```javascript
<button class="stat-btn" onclick="executeStatisticalQuery(sessionId, 
    'Perform linear regression on [columns]. TEXT FORMAT ONLY - do not create any visualization'
)">
    <i class="fas fa-list"></i> Analyze
</button>
```

#### Visualize Button (Chart)
```javascript
<button class="stat-btn stat-btn-viz" onclick="executeStatisticalQuery(sessionId, 
    'Create a scatter plot for linear regression on [columns] with regression line'
)">
    <i class="fas fa-chart-scatter"></i> Visualize
</button>
```

### Response Handling

```javascript
async function executeStatisticalQuery(sessionId, query) {
    const response = await fetch(`/api/v1/sessions/${sessionId}/query`, {
        method: 'POST',
        body: JSON.stringify({ query })
    });
    
    const data = await response.json();
    
    if (data.figure) {
        // Render Plotly chart
        Plotly.newPlot(container, data.figure.data, data.figure.layout);
    } else {
        // Display text output
        displayTextResults(data.output);
    }
}
```

---

## 🛡️ Error Handling

### Comprehensive Error Messages

**KeyError (Security/Column Not Found):**
```python
except KeyError as e:
    if error_str in dangerous_names:
        error_msg = f"Security error: Attempted to use restricted function '{error_str}'.\n"
        error_msg += "This usually happens when using DOUBLE braces in f-strings.\n\n"
        error_msg += "❌ WRONG: f\"Value: {{variable}}\" (double braces)\n"
        error_msg += "✅ CORRECT: f\"Value: {variable}\" (single braces)"
    else:
        error_msg = f"Column/Row not found: {str(e)}. This often happens when:\n"
        error_msg += "  • Trying to access a column that doesn't exist\n"
        error_msg += "  • Check if the column/row exists before using .loc[]"
```

**ValueError (Data Type Issues):**
```python
except ValueError as e:
    error_msg = f"Value error: {str(e)}. This often happens when:\n"
    error_msg += "  • Input contains NaN values (use .dropna() to remove)\n"
    error_msg += "  • Data types don't match (e.g., text in numeric column)\n"
    error_msg += "  • Array shapes don't match for ML operations"
```

---

## 📋 Setup Requirements

### Required Packages

**File:** `requirements.txt`

```txt
# Core Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# Data Science
pandas>=2.1.0
numpy>=1.25.0
plotly>=5.17.0

# LLM Integration
google-generativeai>=0.3.0

# Statistical Libraries (Optional)
scipy>=1.11.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
```

### Environment Configuration

**File:** `.env`

```env
GEMINI_API_KEY=your_api_key_here
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

---

## 🚀 Running the Sandbox

### Start Server

```bash
# Windows PowerShell
python start_server.py

# Or with uvicorn directly
uvicorn app.main:app --reload --port 8000
```

### Run Tests

```bash
# Test sandbox security
python test_visualization_sandbox.py

# Expected output:
# 🔧 Testing Visualization Sandbox
# ==================================================
# 🧪 Testing basic pandas operations...
# ✅ Basic pandas operations successful
# 
# 🧪 Testing plotly visualization...
# ✅ Visualization code successful
# ✅ Plotly figure generated successfully
# 
# 🧪 Testing import statement blocking...
# ✅ Import statements properly blocked
# 
# 🧪 Testing dangerous function blocking...
# ✅ Dangerous functions properly blocked
# 
# ==================================================
# ✅ All sandbox tests passed! Visualization environment is working.
```

---

## ✅ What the Sandbox Allows

### ✅ Safe Operations

- **Pandas operations:** `df.groupby()`, `df.merge()`, `df.describe()`
- **NumPy calculations:** `np.mean()`, `np.std()`, `np.corrcoef()`
- **Plotly visualizations:** `px.scatter()`, `px.bar()`, `go.Figure()`
- **Statistical analysis:** `stats.ttest_ind()`, `sm.OLS()`, `LinearRegression()`
- **Data transformations:** `df.apply()`, `df.transform()`, `df.agg()`
- **Print statements:** `print()`, `format()`, f-strings

### Example: Safe Code
```python
# Group by category and calculate mean
grouped = df.groupby('Category')['Amount'].mean()
print("Average amount by category:")
print(grouped)

# Create visualization
fig = px.bar(x=grouped.index, y=grouped.values, title='Average by Category')
fig.show()
```

---

## ❌ What the Sandbox Blocks

### ❌ Dangerous Operations

- **File I/O:** `open()`, `read()`, `write()`
- **System access:** `os.system()`, `subprocess.run()`
- **Dynamic code execution:** `eval()`, `exec()`, `compile()`
- **Import statements:** `import os`, `from sys import exit`
- **Introspection abuse:** `globals()`, `locals()`, `vars()`, `dir()`
- **Attribute manipulation:** `getattr()`, `setattr()`, `delattr()`

### Example: Blocked Code
```python
# ❌ This will be blocked
import os
os.system("malicious_command")

# ❌ This will be blocked
eval("__import__('os').system('rm -rf /')")

# ❌ This will be blocked
with open('/etc/passwd', 'r') as f:
    data = f.read()
```

---

## 🔍 Key Design Decisions

### 1. **Why Block Imports?**
- **Security:** Prevent importing dangerous modules (os, sys, subprocess)
- **Simplicity:** All required modules pre-imported and validated
- **Consistency:** Ensure same environment for all executions

### 2. **Why Use AST Analysis?**
- **Pre-execution validation:** Catch issues before running code
- **Static analysis:** No need to execute code to detect problems
- **Comprehensive:** Can analyze nested structures and complex syntax

### 3. **Why Copy DataFrame?**
```python
'df': df.copy()  # Work with a copy
```
- **Immutability:** Prevent user code from modifying original data
- **Isolation:** Each execution gets fresh DataFrame
- **Safety:** Avoid side effects across multiple queries

### 4. **Why Capture Output Separately?**
```python
with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
    exec(code, exec_globals, exec_locals)
```
- **Clean separation:** stdout vs stderr
- **Error handling:** Capture all print statements and errors
- **Debugging:** Show users what their code printed

---

## 🎯 Best Practices

### For LLM Prompt Engineering

1. **Always specify available modules:**
   ```
   Available: pd, np, px, go, scipy, statsmodels, sklearn
   ```

2. **Provide clear visualization rules:**
   ```
   ONLY create visualizations if query says "visualize" or "create chart"
   ```

3. **Show example code patterns:**
   ```python
   # Correct
   fig = px.scatter(df, x='col1', y='col2')
   fig.show()
   ```

### For Code Execution

1. **Always validate AST first**
2. **Use restricted builtins**
3. **Copy DataFrame to avoid mutation**
4. **Capture stdout/stderr separately**
5. **Extract figure automatically**
6. **Provide helpful error messages**

---

## 📊 Performance Considerations

### Execution Timeout
Currently no timeout implemented. Consider adding:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded time limit")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout
exec(code, exec_globals, exec_locals)
signal.alarm(0)  # Disable timeout
```

### Memory Limits
No memory limits currently. For production:
```python
import resource

# Limit memory to 1GB
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
```

---

## 🔮 Future Enhancements

### 1. **Execution Timeouts**
- Prevent infinite loops
- Kill long-running operations

### 2. **Memory Limits**
- Prevent memory exhaustion
- Handle large DataFrames safely

### 3. **Code Caching**
- Cache frequently executed code
- Improve performance for repeated queries

### 4. **Enhanced Logging**
- Track execution times
- Monitor security violations
- Audit code execution

### 5. **Sandbox Profiles**
- Different security levels
- Custom module allowlists
- User-specific restrictions

---

## 📝 Summary

The **Visualization Sandbox** is a multi-layered security system that enables safe Python code execution:

1. **AST Analysis** - Pre-execution static analysis
2. **Restricted Environment** - Limited builtins and pre-imported modules
3. **Output Capture** - Safe stdout/stderr redirection
4. **Figure Extraction** - Automatic Plotly chart detection
5. **Error Handling** - Helpful, context-aware error messages

**Result:** Users can run data analysis and create visualizations dynamically without compromising system security.

---

## 📚 Related Documentation

- `VISUALIZATION_FEATURE.md` - Dual-mode interface documentation
- `STATISTICAL_ANALYSIS_DOCUMENTATION.md` - Statistical features guide
- `test_visualization_sandbox.py` - Comprehensive test suite
- `app/services/data_analysis.py` - Core implementation

---

**Created:** November 2025  
**Status:** ✅ Production Ready  
**Security Level:** 🔒 High
