# Advanced Statistical Analysis Suite - Documentation

## 📊 Overview

The Advanced Statistical Analysis Suite provides professional-grade statistical testing, regression analysis, outlier detection, and time series analysis capabilities to the PyData Backend.

## 🚀 Features Implemented

### 1. **Hypothesis Testing**
Professional statistical tests with p-values and interpretations

#### Available Tests:

##### Normality Tests
- **Shapiro-Wilk Test** - Best for n < 5000
- **D'Agostino-Pearson Test** - Based on skewness and kurtosis
- **Anderson-Darling Test** - General purpose
- **Kolmogorov-Smirnov Test** - Compare to theoretical distribution

**Endpoint:** `POST /api/v1/statistical-analysis/normality-test`

**Example Request:**
```json
{
  "session_id": "abc123",
  "column": "sales",
  "alpha": 0.05,
  "methods": ["shapiro", "dagostino", "anderson", "ks"]
}
```

**Example Response:**
```json
{
  "success": true,
  "test_results": {
    "shapiro_wilk": {
      "test_name": "Shapiro-Wilk Test",
      "statistic": 0.9876,
      "p_value": 0.1234,
      "significant": false,
      "interpretation": "Data is normally distributed (p=0.1234)",
      "confidence_level": 0.95,
      "additional_info": {
        "sample_size": 1000,
        "recommended_for": "n < 5000"
      }
    }
  },
  "message": "Normality tests completed for column 'sales'",
  "execution_time": 0.234
}
```

---

##### T-Tests
- **Independent Samples T-Test** - Compare means of two independent groups
- **Welch's T-Test** - When variances are unequal
- **Paired Samples T-Test** - Compare before/after measurements

**Endpoint:** `POST /api/v1/statistical-analysis/t-test`

**Example - Independent T-Test:**
```json
{
  "session_id": "abc123",
  "test_type": "independent",
  "group_col": "treatment",
  "value_col": "outcome",
  "group1_value": "control",
  "group2_value": "experimental",
  "alpha": 0.05,
  "equal_var": true
}
```

**Example - Paired T-Test:**
```json
{
  "session_id": "abc123",
  "test_type": "paired",
  "before_col": "pre_test",
  "after_col": "post_test",
  "alpha": 0.05
}
```

**Response includes:**
- Test statistic and p-value
- Mean and standard deviation for each group
- Effect size (Cohen's d)
- Interpretation

---

##### ANOVA (Analysis of Variance)
Compare means across multiple groups

**Endpoint:** `POST /api/v1/statistical-analysis/anova`

```json
{
  "session_id": "abc123",
  "group_col": "department",
  "value_col": "salary",
  "alpha": 0.05
}
```

**Response includes:**
- F-statistic and p-value
- Group means and sizes
- Effect size (eta-squared)
- Interpretation

---

##### Chi-Square Test
Test independence between categorical variables

**Endpoint:** `POST /api/v1/statistical-analysis/chi-square`

```json
{
  "session_id": "abc123",
  "col1": "gender",
  "col2": "product_preference",
  "alpha": 0.05
}
```

**Response includes:**
- Chi-square statistic and p-value
- Degrees of freedom
- Contingency table
- Cramér's V effect size

---

##### Correlation Tests
- **Pearson** - Linear relationships (assumes normality)
- **Spearman** - Monotonic relationships (rank-based)
- **Kendall Tau** - Ordinal associations

**Endpoint:** `POST /api/v1/statistical-analysis/correlation-test`

```json
{
  "session_id": "abc123",
  "col1": "advertising_spend",
  "col2": "revenue",
  "method": "pearson",
  "alpha": 0.05
}
```

---

### 2. **Outlier Detection**

Multiple methods for robust outlier identification

#### Available Methods:

##### IQR (Interquartile Range) Method
Classic and robust method using quartiles

```json
{
  "session_id": "abc123",
  "column": "price",
  "method": "iqr",
  "iqr_multiplier": 1.5
}
```

##### Z-Score Method
Based on standard deviations from mean

```json
{
  "session_id": "abc123",
  "column": "price",
  "method": "zscore",
  "zscore_threshold": 3.0
}
```

##### Modified Z-Score (MAD)
More robust to outliers than standard Z-score

```json
{
  "session_id": "abc123",
  "column": "price",
  "method": "modified_zscore",
  "modified_zscore_threshold": 3.5
}
```

##### Isolation Forest
Machine learning-based anomaly detection

```json
{
  "session_id": "abc123",
  "column": "price",
  "method": "isolation_forest",
  "isolation_contamination": 0.1
}
```

##### All Methods at Once
```json
{
  "session_id": "abc123",
  "column": "price",
  "method": "all"
}
```

**Endpoint:** `POST /api/v1/statistical-analysis/outlier-detection`

**Response includes:**
- Outlier indices and values
- Count and percentage
- Method-specific bounds/thresholds
- Comparison across methods (when using "all")

---

### 3. **Regression Analysis**

Professional regression with comprehensive metrics

#### Linear Regression
Single or multiple independent variables

```json
{
  "session_id": "abc123",
  "regression_type": "linear",
  "x_col": ["feature1", "feature2", "feature3"],
  "y_col": "target"
}
```

**Response includes:**
- R² and Adjusted R²
- Coefficients and intercept
- RMSE and MAE
- Feature importance (standardized coefficients)
- Predictions and residuals

#### Polynomial Regression
Fit non-linear relationships

```json
{
  "session_id": "abc123",
  "regression_type": "polynomial",
  "x_col": "temperature",
  "y_col": "ice_cream_sales",
  "polynomial_degree": 2
}
```

#### Logistic Regression
Binary classification

```json
{
  "session_id": "abc123",
  "regression_type": "logistic",
  "x_col": ["age", "income", "education"],
  "y_col": "purchased"
}
```

**Response includes:**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Coefficients
- Predicted probabilities

**Endpoint:** `POST /api/v1/statistical-analysis/regression`

---

### 4. **Distribution Fitting**

Automatically find the best probability distribution

**Available Distributions:**
- Normal (Gaussian)
- Exponential
- Gamma
- Log-normal
- Weibull

```json
{
  "session_id": "abc123",
  "column": "response_time",
  "distributions": ["norm", "expon", "gamma", "lognorm", "weibull_min"]
}
```

**Endpoint:** `POST /api/v1/statistical-analysis/distribution-fit`

**Response includes:**
- Distribution parameters
- Kolmogorov-Smirnov test results
- AIC and BIC for model comparison
- Best fit identification
- Goodness-of-fit p-values

---

### 5. **Time Series Analysis**

#### Stationarity Test (Augmented Dickey-Fuller)

```json
{
  "session_id": "abc123",
  "column": "stock_price",
  "alpha": 0.05
}
```

**Endpoint:** `POST /api/v1/statistical-analysis/stationarity-test`

**Response includes:**
- ADF statistic and p-value
- Critical values at different significance levels
- Stationarity determination
- Interpretation

---

#### Granger Causality Test
Test if one time series predicts another

```json
{
  "session_id": "abc123",
  "cause_col": "advertising",
  "effect_col": "sales",
  "max_lag": 5,
  "alpha": 0.05
}
```

**Endpoint:** `POST /api/v1/statistical-analysis/granger-causality`

**Response includes:**
- Test results for each lag
- F-statistics and p-values
- Significant lags identified
- Causality interpretation

---

### 6. **Summary Statistics**

Comprehensive statistical summary

```json
{
  "session_id": "abc123",
  "column": "revenue"
}
```

**Endpoint:** `POST /api/v1/statistical-analysis/summary-statistics`

**Response includes:**
- Count, Mean, Std, Min, Max
- Quartiles (Q1, Median, Q3)
- Skewness and Kurtosis
- Range, IQR, Variance
- Coefficient of Variation

---

## 📦 Installation

### Required Packages

Add to `requirements.txt`:
```
scipy>=1.11.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
prophet>=1.1.5  # For future time series forecasting
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"
session_id = "your_session_id"

# Normality Test
response = requests.post(
    f"{BASE_URL}/statistical-analysis/normality-test",
    json={
        "session_id": session_id,
        "column": "age",
        "alpha": 0.05
    }
)
print(response.json())

# Outlier Detection
response = requests.post(
    f"{BASE_URL}/statistical-analysis/outlier-detection",
    json={
        "session_id": session_id,
        "column": "price",
        "method": "all"
    }
)
outliers = response.json()

# Linear Regression
response = requests.post(
    f"{BASE_URL}/statistical-analysis/regression",
    json={
        "session_id": session_id,
        "regression_type": "linear",
        "x_col": ["square_feet", "bedrooms"],
        "y_col": "price"
    }
)
regression_results = response.json()
```

### JavaScript/Frontend

```javascript
// Perform T-Test
const tTestResult = await fetch('/api/v1/statistical-analysis/t-test', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: sessionId,
    test_type: 'independent',
    group_col: 'treatment',
    value_col: 'outcome',
    group1_value: 'A',
    group2_value: 'B',
    alpha: 0.05
  })
});

const result = await tTestResult.json();
console.log(result.test_results);

// Distribution Fitting
const distFit = await fetch('/api/v1/statistical-analysis/distribution-fit', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: sessionId,
    column: 'response_time'
  })
});

const fitResults = await distFit.json();
console.log('Best distribution:', fitResults.best_distribution);
```

---

## 🔧 Integration with LLM

The statistical analysis capabilities are also available through natural language queries:

**User:** "Test if sales data is normally distributed"

**System:** Uses normality tests and provides interpretation

**User:** "Are there any outliers in the price column?"

**System:** Applies multiple outlier detection methods

**User:** "Is there a relationship between advertising and revenue?"

**System:** Performs correlation test with significance testing

**User:** "Predict sales based on marketing spend and seasonality"

**System:** Performs regression analysis with detailed metrics

---

## 📊 Statistical Interpretations

All tests provide:
- ✅ **Automatic Interpretation** - Plain English explanations
- 📈 **Effect Sizes** - Practical significance (Cohen's d, Cramér's V, eta-squared)
- 🎯 **Confidence Levels** - User-specified alpha levels
- 📉 **Visual Recommendations** - Suggestions for follow-up analyses
- ⚠️ **Assumption Checks** - Warnings about violations

---

## 🎓 Best Practices

### When to Use Each Test

**Normality Tests:**
- Before parametric tests (t-tests, ANOVA, regression)
- Sample size < 5000: Use Shapiro-Wilk
- Sample size > 5000: Use D'Agostino-Pearson

**T-Tests:**
- Independent: Compare two unrelated groups
- Paired: Before/after measurements on same subjects
- Welch's: When group variances differ significantly

**ANOVA:**
- Compare 3+ groups
- Follow up with post-hoc tests if significant

**Correlation:**
- Pearson: Linear relationships, normal data
- Spearman: Monotonic relationships, ordinal data
- Kendall: Small samples, many tied ranks

**Outlier Detection:**
- IQR: Robust, good for skewed data
- Z-score: Assumes normality
- Modified Z-score: Better for datasets with outliers
- Isolation Forest: Complex multivariate patterns

---

## 🚀 Future Enhancements

Planned features:
- ✅ ARIMA/Prophet time series forecasting
- ✅ Bayesian A/B testing
- ✅ Multiple comparison corrections (Bonferroni, FDR)
- ✅ Power analysis and sample size calculations
- ✅ Bootstrap confidence intervals
- ✅ Mixed-effects models
- ✅ Survival analysis

---

## 📞 Support

For issues or questions:
- Check API documentation at `/docs`
- Review test results for detailed diagnostics
- Ensure data meets test assumptions
- Use appropriate alpha levels (typically 0.05 or 0.01)

---

## 📝 License

Part of PyData Assistant Backend - Professional Data Analysis Platform

---

**Created:** October 2025
**Version:** 1.0.0
**Status:** Production Ready ✅
