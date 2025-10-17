#!/usr/bin/env python3
"""
Test script to verify query response validation fixes
"""

import pandas as pd
from pydantic import ValidationError
from app.services.data_analysis import convert_numpy_types, QueryResponseData
from app.api.v1.schemas.session import QueryResponse, ErrorData, TextData

def test_error_response_validation():
    """Test that error responses validate correctly"""
    print("🧪 Testing error response validation...")
    
    # Create error response data like what the service generates
    error_data = {
        'response_type': 'error',
        'error': 'Test error message',
        'message': 'Code execution failed', 
        'error_type': 'execution_error'
    }
    
    try:
        # Test ErrorData validation directly
        error_model = ErrorData(**error_data)
        print(f"✅ ErrorData validation passed: {error_model}")
        
        # Test QueryResponse with error data
        response = QueryResponse(
            session_id='test-session',
            query='test query',
            response_type='error',
            data=error_data,
            message='Test error',
            execution_time=1.0
        )
        print(f"✅ QueryResponse validation passed: {response.response_type}")
        return True
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
        return False

def test_text_response_validation():
    """Test that text responses validate correctly"""  
    print("\n🧪 Testing text response validation...")
    
    # Create text response data
    text_data = {
        'response_type': 'text',
        'text': 'This is a text response',
        'message': 'Analysis response'
    }
    
    try:
        # Test TextData validation directly
        text_model = TextData(**text_data)
        print(f"✅ TextData validation passed: {text_model}")
        
        # Test QueryResponse with text data
        response = QueryResponse(
            session_id='test-session',
            query='test query',
            response_type='text',
            data=text_data,
            message='Test text',
            execution_time=1.0
        )
        print(f"✅ QueryResponse validation passed: {response.response_type}")
        return True
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
        return False

def test_statistics_response_validation():
    """Test that statistics responses validate correctly"""
    print("\n🧪 Testing statistics response validation...")
    
    # Create mock DataFrame
    df = pd.DataFrame({
        'numbers': [1, 2, 3, 4, 5],
        'values': [10, 20, 30, 40, 50]
    })
    
    # Create statistics response data with NumPy types
    stats_data = {
        'response_type': 'statistics',
        'calculation': 'Statistical Analysis',
        'results': {
            'mean': df['numbers'].mean(),  # This will be numpy.float64
            'std': df['numbers'].std(),    # This will be numpy.float64
        },
        'interpretation': 'Test statistics output',
        'summary_stats': {
            'numbers_mean': df['numbers'].mean(),
            'numbers_std': df['numbers'].std()
        }
    }
    
    try:
        # Convert NumPy types first
        converted_data = convert_numpy_types(stats_data)
        print(f"✅ NumPy conversion successful")
        
        # Test QueryResponse with statistics data
        response = QueryResponse(
            session_id='test-session',
            query='show statistics',
            response_type='statistics',
            data=converted_data,
            message='Statistics generated',
            execution_time=2.0
        )
        print(f"✅ QueryResponse validation passed: {response.response_type}")
        return True
        
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Query Response Validation Fixes")
    print("=" * 50)
    
    test1_passed = test_error_response_validation()
    test2_passed = test_text_response_validation() 
    test3_passed = test_statistics_response_validation()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed and test3_passed:
        print("✅ All validation tests passed! Schema fixes are working.")
    else:
        print("❌ Some validation tests failed. Check output above.")