#!/usr/bin/env python3
"""
Test script to verify NumPy JSON serialization fix
"""

import json
import pandas as pd
import numpy as np
from app.services.data_analysis import convert_numpy_types, get_data_preview

def test_numpy_conversion():
    """Test NumPy type conversion"""
    print("🧪 Testing NumPy type conversion...")
    
    # Create test data with NumPy types
    test_data = {
        'int64_val': np.int64(42),
        'float64_val': np.float64(3.14),
        'bool_val': np.bool_(True),
        'array_val': np.array([1, 2, 3]),
        'nested': {
            'inner_int': np.int32(100),
            'inner_float': np.float32(2.5)
        }
    }
    
    print("Original data types:")
    for key, val in test_data.items():
        print(f"  {key}: {type(val)} = {val}")
    
    # Convert using our function
    converted = convert_numpy_types(test_data)
    
    print("\nConverted data types:")
    for key, val in converted.items():
        print(f"  {key}: {type(val)} = {val}")
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(converted)
        print("\n✅ JSON serialization successful!")
        print(f"JSON string: {json_str}")
        return True
    except Exception as e:
        print(f"\n❌ JSON serialization failed: {e}")
        return False

def test_data_preview():
    """Test data preview with NumPy types"""
    print("\n🧪 Testing data preview with NumPy types...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'integers': [1, 2, 3, 4, 5],
        'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
        'strings': ['a', 'b', 'c', 'd', 'e'],
        'booleans': [True, False, True, False, True]
    })
    
    # Get preview
    try:
        preview = get_data_preview(df, 'test.csv')
        
        # Try to serialize
        json_str = json.dumps(preview)
        print("✅ Data preview serialization successful!")
        print(f"Preview contains: {list(preview.keys())}")
        return True
    except Exception as e:
        print(f"❌ Data preview serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing NumPy JSON Serialization Fix")
    print("=" * 50)
    
    test1_passed = test_numpy_conversion()
    test2_passed = test_data_preview()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("✅ All tests passed! NumPy fix is working.")
    else:
        print("❌ Some tests failed. Check the output above.")