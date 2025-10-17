#!/usr/bin/env python3
"""
Test specific visualization scenarios that were failing
"""

import pandas as pd
from app.services.data_analysis import _execute_code_safely

def test_item_description_visualization():
    """Test the specific Item Description visualization that was failing"""
    print("🎯 Testing Item Description visualization (the failing case)...")
    
    # Create test DataFrame similar to user's data
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02'],
        'Customer Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David'],
        'Item Description': ['Apple', 'Banana', 'Orange', 'Apple', 'Grapes'],
        'Quantity': [1, 2, None, 4, 5],  # Some missing values
        'Amount (₹)': [10.5, 20.0, 15.5, None, 22.5],  # Some missing values
        'Remarks': [None, 'Good', None, 'Fresh', None]  # Many missing values
    })
    
    # Test the kind of code that should be generated for "Create a visualization of item description"
    code = """
# Analyze Item Description column
item_counts = df['Item Description'].value_counts()
print("Item Description Distribution:")
print(item_counts)
print()

# Create visualization
fig = px.bar(
    x=item_counts.index,
    y=item_counts.values,
    title='Distribution of Item Descriptions',
    labels={'x': 'Item Description', 'y': 'Count'}
)

# Show the figure (this will be captured)
fig.show()
"""
    
    result = _execute_code_safely(code, df)
    
    if result['success']:
        print("✅ Item Description visualization successful!")
        print(f"Console output:\n{result['output']}")
        
        if result.get('figure'):
            print("✅ Plotly figure generated and captured!")
            # Print some figure details
            fig_data = result['figure']
            if 'data' in fig_data and fig_data['data']:
                trace = fig_data['data'][0]
                print(f"Chart type: {trace.get('type', 'unknown')}")
                print(f"X data points: {len(trace.get('x', []))}")
                print(f"Y data points: {len(trace.get('y', []))}")
        else:
            print("⚠️ No plotly figure found - this might indicate an issue")
        
        return True
    else:
        print(f"❌ Visualization failed: {result['error']}")
        print(f"Output so far: {result.get('output', 'No output')}")
        return False

def test_missing_values_analysis():
    """Test missing values analysis that was working"""
    print("\n🎯 Testing missing values analysis (the working case)...")
    
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02'],
        'Customer Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David'],
        'Item Description': ['Apple', 'Banana', 'Orange', 'Apple', 'Grapes'],
        'Quantity': [1, 2, None, 4, 5],
        'Amount (₹)': [10.5, 20.0, 15.5, None, 22.5],
        'Remarks': [None, 'Good', None, 'Fresh', None]
    })
    
    code = """
# Calculate missing values
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)

print("Missing Values Information:")
print("Missing Count | Missing Percentage (%)")

for col in df.columns:
    count = missing_count[col]
    percentage = missing_percentage[col]
    print(f"{col}: {count} ({percentage:.6f}%)")
"""
    
    result = _execute_code_safely(code, df)
    
    if result['success']:
        print("✅ Missing values analysis successful!")
        print(f"Output:\n{result['output']}")
        return True
    else:
        print(f"❌ Missing values analysis failed: {result['error']}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Specific Failing Scenarios")
    print("=" * 60)
    
    test1_passed = test_item_description_visualization()
    test2_passed = test_missing_values_analysis()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✅ All specific scenario tests passed!")
        print("The visualization sandbox should now work correctly.")
    else:
        print("❌ Some tests failed - issues may persist.")