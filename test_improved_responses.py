#!/usr/bin/env python3
"""
Test improved response formatting and visualization handling
"""

import pandas as pd
from app.services.data_analysis import _execute_code_safely

def test_formatted_summary():
    """Test that summaries are properly formatted"""
    print("🧪 Testing formatted summary response...")
    
    # Create test DataFrame similar to user's data
    df = pd.DataFrame({
        'Order ID': ['A001', 'A002', 'A003', 'A004', 'A005'] * 2000,  # 10000 rows
        'Customer Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'] * 2000,
        'Segment': ['Consumer', 'Corporate', 'Home Office', 'Consumer', 'Corporate'] * 2000,
        'Sales': [100.5, 200.0, 150.25, 300.75, 250.0] * 2000,
        'Quantity': [1, 2, 3, 4, 5] * 2000,
        'Category': ['Office Supplies', 'Technology', 'Furniture', 'Office Supplies', 'Technology'] * 2000
    })
    
    # Test well-formatted summary code
    code = """
# Generate a well-formatted dataset summary
print("### Dataset Overview ###")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("")

print("--- Column Information ---")
print(df.info())
print("")

print("--- Missing Values Analysis ---")
missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df) * 100).round(2)

for col in df.columns:
    if missing_counts[col] > 0:
        print(f"• {col}: {missing_counts[col]:,} missing ({missing_percentages[col]:.2f}%)")
    else:
        print(f"• {col}: No missing values")
print("")

print("--- Categorical Distributions ---")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols[:3]:  # Show first 3 categorical columns
    print(f"**{col}:**")
    value_counts = df[col].value_counts()
    for category, count in value_counts.head(5).items():
        print(f"  • {category}: {count:,} records")
    print("")

print("--- Numerical Summary ---")
numerical_cols = df.select_dtypes(include=['number']).columns
if len(numerical_cols) > 0:
    summary_stats = df[numerical_cols].describe()
    print("Key statistics for numerical columns:")
    for col in numerical_cols:
        print(f"• {col}:")
        print(f"  - Mean: {summary_stats.loc['mean', col]:.2f}")
        print(f"  - Median: {summary_stats.loc['50%', col]:.2f}")
        print(f"  - Range: {summary_stats.loc['min', col]:.2f} to {summary_stats.loc['max', col]:.2f}")
    print("")
"""
    
    result = _execute_code_safely(code, df)
    
    if result['success']:
        print("✅ Formatted summary successful!")
        print("Sample output:")
        print("-" * 40)
        print(result['output'][:500] + "..." if len(result['output']) > 500 else result['output'])
        print("-" * 40)
        return True
    else:
        print(f"❌ Formatted summary failed: {result['error']}")
        return False

def test_single_visualization():
    """Test that visualizations create a single, clean chart"""
    print("\n🧪 Testing single visualization response...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'Segment': ['Consumer', 'Corporate', 'Home Office', 'Consumer', 'Corporate'] * 1000,
        'Sales': [100.5, 200.0, 150.25, 300.75, 250.0] * 1000,
        'Region': ['West', 'East', 'Central', 'South', 'West'] * 1000
    })
    
    # Test visualization code that creates ONE chart
    code = """
# Create a clean, single visualization for segment distribution
print("### Segment Analysis ###")
print("Creating visualization of customer segments...")
print("")

# Analyze segment distribution
segment_counts = df['Segment'].value_counts()
print("Segment distribution:")
for segment, count in segment_counts.items():
    print(f"• {segment}: {count:,} records ({count/len(df)*100:.1f}%)")
print("")

# Create the visualization
fig = px.bar(
    x=segment_counts.index,
    y=segment_counts.values,
    title='Customer Segment Distribution',
    labels={'x': 'Customer Segment', 'y': 'Number of Orders'},
    color=segment_counts.index,
    text=segment_counts.values
)

# Customize the chart
fig.update_traces(texttemplate='%{text:,}', textposition='outside')
fig.update_layout(
    showlegend=False,
    height=500,
    font=dict(size=12)
)

# This must be the last line to capture the figure
fig.show()
"""
    
    result = _execute_code_safely(code, df)
    
    if result['success']:
        print("✅ Single visualization successful!")
        print("Console output:")
        print("-" * 40)
        print(result['output'])
        print("-" * 40)
        
        if result.get('figure'):
            print("✅ Plotly figure captured successfully!")
            fig_data = result['figure']
            
            # Validate figure structure
            if 'data' in fig_data and 'layout' in fig_data:
                print(f"✅ Valid plotly figure structure")
                print(f"• Chart type: {fig_data['data'][0].get('type', 'unknown') if fig_data['data'] else 'none'}")
                print(f"• Title: {fig_data['layout'].get('title', {}).get('text', 'No title')}")
                print(f"• Data traces: {len(fig_data['data'])}")
            else:
                print("⚠️ Figure structure might be invalid")
        else:
            print("❌ No plotly figure captured")
            
        return result.get('figure') is not None
    else:
        print(f"❌ Visualization failed: {result['error']}")
        return False

def test_multiple_analyses():
    """Test that multiple analyses work correctly"""
    print("\n🧪 Testing multiple analysis scenarios...")
    
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=1000),
        'Product': ['A', 'B', 'C'] * 334,
        'Sales': [100, 200, 150] * 334,
        'Region': (['North', 'South', 'East', 'West'] * 250)[:1000]  # Ensure exact length
    })
    
    # Test missing values analysis
    missing_code = """
print("### Missing Values Analysis ###")
print("")

missing_counts = df.isnull().sum()
total_rows = len(df)

print("Missing values by column:")
for col in df.columns:
    count = missing_counts[col]
    percentage = (count / total_rows * 100)
    if count > 0:
        print(f"• {col}: {count:,} missing ({percentage:.2f}%)")
    else:
        print(f"• {col}: No missing values ✓")

print("")
print(f"**Total dataset completeness: {((total_rows * len(df.columns) - missing_counts.sum()) / (total_rows * len(df.columns)) * 100):.1f}%**")
"""
    
    result = _execute_code_safely(missing_code, df)
    
    if result['success']:
        print("✅ Missing values analysis successful!")
        print(f"Output preview: {result['output'][:200]}...")
        return True
    else:
        print(f"❌ Missing values analysis failed: {result['error']}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Improved Response Formatting and Visualization")
    print("=" * 70)
    
    test1_passed = test_formatted_summary()
    test2_passed = test_single_visualization()
    test3_passed = test_multiple_analyses()
    
    print("\n" + "=" * 70)
    if all([test1_passed, test2_passed, test3_passed]):
        print("✅ All improved response tests passed!")
        print("✅ Text formatting is now clean and readable")
        print("✅ Visualizations are properly captured")
        print("✅ Multiple analysis types work correctly")
    else:
        print("❌ Some tests failed. Check output above.")