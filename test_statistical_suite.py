"""
Test script for Advanced Statistical Analysis Suite
Run this after starting the server to validate all features
"""

import requests
import pandas as pd
import numpy as np
import json

BASE_URL = "http://localhost:8000/api/v1"

# Create test data
np.random.seed(42)
test_data = pd.DataFrame({
    'normal_data': np.random.normal(100, 15, 1000),
    'skewed_data': np.random.exponential(2, 1000),
    'group': np.random.choice(['A', 'B', 'C'], 1000),
    'category1': np.random.choice(['X', 'Y'], 1000),
    'category2': np.random.choice(['P', 'Q'], 1000),
    'feature1': np.random.uniform(0, 100, 1000),
    'feature2': np.random.uniform(0, 50, 1000),
    'target': np.random.uniform(0, 200, 1000)
})

# Add some outliers
test_data.loc[np.random.choice(1000, 10, replace=False), 'normal_data'] = np.random.uniform(200, 300, 10)

# Save test data
test_data.to_csv('test_statistical_data.csv', index=False)

print("=" * 80)
print("🧪 Testing Advanced Statistical Analysis Suite")
print("=" * 80)


def test_endpoint(name, endpoint, payload):
    """Test an endpoint and display results."""
    print(f"\n📊 Testing: {name}")
    print("-" * 80)
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Execution Time: {result.get('execution_time', 'N/A'):.3f}s")
            print(f"Message: {result.get('message', 'N/A')}")
            
            # Display key results
            if 'test_results' in result:
                for test_name, test_result in result['test_results'].items():
                    if isinstance(test_result, dict) and 'interpretation' in test_result:
                        print(f"  • {test_result.get('test_name', test_name)}: {test_result.get('interpretation')}")
            
            if 'outlier_results' in result:
                for method, outlier_result in result['outlier_results'].items():
                    if isinstance(outlier_result, dict):
                        count = outlier_result.get('outlier_count', 0)
                        percentage = outlier_result.get('outlier_percentage', 0)
                        print(f"  • {outlier_result.get('method', method)}: {count} outliers ({percentage:.2f}%)")
            
            if 'regression_results' in result:
                reg = result['regression_results']
                if 'r_squared' in reg:
                    print(f"  • R² = {reg['r_squared']:.4f}")
                    print(f"  • RMSE = {reg.get('rmse', 'N/A'):.4f}")
                if 'accuracy' in reg:
                    print(f"  • Accuracy = {reg['accuracy']:.4f}")
            
            if 'fit_results' in result:
                print(f"  • Best Distribution: {result.get('best_distribution', 'N/A')}")
            
            if 'analysis_results' in result:
                analysis = result['analysis_results']
                if 'interpretation' in analysis:
                    print(f"  • {analysis['interpretation']}")
            
            if 'statistics' in result:
                stats = result['statistics']
                print(f"  • Mean: {stats.get('mean', 'N/A'):.2f}")
                print(f"  • Std: {stats.get('std', 'N/A'):.2f}")
                print(f"  • Skewness: {stats.get('skewness', 'N/A'):.2f}")
            
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


# First, create a session by uploading the test data
print("\n🔄 Step 1: Creating session with test data...")
print("-" * 80)

with open('test_statistical_data.csv', 'rb') as f:
    files = {'file': ('test_statistical_data.csv', f, 'text/csv')}
    response = requests.post(f"{BASE_URL}/session/start", files=files)

if response.status_code == 200:
    session_data = response.json()
    session_id = session_data['session_id']
    print(f"✅ Session created: {session_id}")
else:
    print(f"❌ Failed to create session: {response.status_code}")
    print(response.text)
    exit(1)


# Test all endpoints
tests = [
    {
        "name": "Normality Test (Multiple Methods)",
        "endpoint": "/statistical-analysis/normality-test",
        "payload": {
            "session_id": session_id,
            "column": "normal_data",
            "alpha": 0.05,
            "methods": ["shapiro", "dagostino", "anderson"]
        }
    },
    {
        "name": "Independent T-Test",
        "endpoint": "/statistical-analysis/t-test",
        "payload": {
            "session_id": session_id,
            "test_type": "independent",
            "group_col": "group",
            "value_col": "normal_data",
            "group1_value": "A",
            "group2_value": "B",
            "alpha": 0.05
        }
    },
    {
        "name": "ANOVA (One-Way)",
        "endpoint": "/statistical-analysis/anova",
        "payload": {
            "session_id": session_id,
            "group_col": "group",
            "value_col": "normal_data",
            "alpha": 0.05
        }
    },
    {
        "name": "Chi-Square Test",
        "endpoint": "/statistical-analysis/chi-square",
        "payload": {
            "session_id": session_id,
            "col1": "category1",
            "col2": "category2",
            "alpha": 0.05
        }
    },
    {
        "name": "Correlation Test (Pearson)",
        "endpoint": "/statistical-analysis/correlation-test",
        "payload": {
            "session_id": session_id,
            "col1": "feature1",
            "col2": "feature2",
            "method": "pearson",
            "alpha": 0.05
        }
    },
    {
        "name": "Outlier Detection (All Methods)",
        "endpoint": "/statistical-analysis/outlier-detection",
        "payload": {
            "session_id": session_id,
            "column": "normal_data",
            "method": "all"
        }
    },
    {
        "name": "Linear Regression",
        "endpoint": "/statistical-analysis/regression",
        "payload": {
            "session_id": session_id,
            "regression_type": "linear",
            "x_col": ["feature1", "feature2"],
            "y_col": "target"
        }
    },
    {
        "name": "Distribution Fitting",
        "endpoint": "/statistical-analysis/distribution-fit",
        "payload": {
            "session_id": session_id,
            "column": "normal_data",
            "distributions": ["norm", "expon", "gamma"]
        }
    },
    {
        "name": "Summary Statistics",
        "endpoint": "/statistical-analysis/summary-statistics",
        "payload": {
            "session_id": session_id,
            "column": "normal_data"
        }
    }
]

# Run all tests
results = []
for test in tests:
    success = test_endpoint(test["name"], test["endpoint"], test["payload"])
    results.append({"test": test["name"], "success": success})

# Summary
print("\n" + "=" * 80)
print("📊 Test Summary")
print("=" * 80)

passed = sum(1 for r in results if r["success"])
total = len(results)

for result in results:
    status = "✅" if result["success"] else "❌"
    print(f"{status} {result['test']}")

print("-" * 80)
print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

if passed == total:
    print("\n🎉 All tests passed! Statistical Analysis Suite is working perfectly!")
else:
    print(f"\n⚠️ {total - passed} test(s) failed. Check the logs above for details.")

print("\n💡 Tip: View full API documentation at http://localhost:8000/docs")
print("=" * 80)
