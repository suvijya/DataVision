#!/usr/bin/env python3
"""
Test script to verify visualization sandbox works properly
"""

import pandas as pd
from app.services.data_analysis import _execute_code_safely

def test_basic_code_execution():
    """Test basic pandas operations"""
    print("🧪 Testing basic pandas operations...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'Item Description': ['Apple', 'Banana', 'Orange', 'Apple', 'Banana'],
        'Quantity': [1, 2, 3, 4, 5],
        'Amount': [10.5, 20.0, 15.5, 12.0, 22.5]
    })
    
    # Test simple pandas code
    code = """
# Check for missing values
missing_info = df.isnull().sum()
print("Missing values per column:")
for col, count in missing_info.items():
    print(f"{col}: {count}")
"""
    
    result = _execute_code_safely(code, df)
    if result['success']:
        print("✅ Basic pandas operations successful")
        print(f"Output: {result['output']}")
        return True
    else:
        print(f"❌ Basic pandas operations failed: {result['error']}")
        return False

def test_visualization_code():
    """Test plotly visualization code"""
    print("\n🧪 Testing plotly visualization...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'Item Description': ['Apple', 'Banana', 'Orange', 'Apple', 'Banana'],
        'Quantity': [1, 2, 3, 4, 5],
        'Amount': [10.5, 20.0, 15.5, 12.0, 22.5]
    })
    
    # Test visualization code (NO IMPORTS)
    code = """
# Create a histogram of Item Description
value_counts = df['Item Description'].value_counts()
print("Item Description counts:")
print(value_counts)

# Create a plotly bar chart
fig = px.bar(
    x=value_counts.index,
    y=value_counts.values,
    title='Distribution of Item Descriptions',
    labels={'x': 'Item Description', 'y': 'Count'}
)
fig.show()
"""
    
    result = _execute_code_safely(code, df)
    if result['success']:
        print("✅ Visualization code successful")
        print(f"Output: {result['output']}")
        if result.get('figure'):
            print("✅ Plotly figure generated successfully")
        else:
            print("⚠️  No plotly figure found in result")
        return True
    else:
        print(f"❌ Visualization code failed: {result['error']}")
        return False

def test_import_blocking():
    """Test that import statements are properly blocked"""
    print("\n🧪 Testing import statement blocking...")
    
    df = pd.DataFrame({'test': [1, 2, 3]})
    
    # Test code with import (should fail)
    code = """
import pandas as pd
print("This should not execute")
"""
    
    result = _execute_code_safely(code, df)
    if not result['success'] and 'import' in result['error'].lower():
        print("✅ Import statements properly blocked")
        print(f"Error: {result['error']}")
        return True
    else:
        print(f"❌ Import blocking failed: {result}")
        return False

def test_dangerous_functions():
    """Test that dangerous functions are blocked"""
    print("\n🧪 Testing dangerous function blocking...")
    
    df = pd.DataFrame({'test': [1, 2, 3]})
    
    # Test dangerous function usage
    code = """
result = eval("1 + 1")
print(result)
"""
    
    result = _execute_code_safely(code, df)
    if not result['success'] and 'not allowed' in result['error']:
        print("✅ Dangerous functions properly blocked")
        print(f"Error: {result['error']}")
        return True
    else:
        print(f"❌ Dangerous function blocking failed: {result}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Visualization Sandbox")
    print("=" * 50)
    
    test1_passed = test_basic_code_execution()
    test2_passed = test_visualization_code()
    test3_passed = test_import_blocking()
    test4_passed = test_dangerous_functions()
    
    print("\n" + "=" * 50)
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("✅ All sandbox tests passed! Visualization environment is working.")
    else:
        print("❌ Some sandbox tests failed. Check output above.")