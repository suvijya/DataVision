# PyData Assistant - Complete Usage Guide

## 🚀 Quick Start

### Starting the Server

**Method 1: Using the Start Script (Recommended)**
```bash
python start_server.py
```

**Method 2: Using Uvicorn Directly**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Method 3: Using Python Module**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will automatically:
- Start on `http://localhost:8000`
- Open your browser to the application
- Display API docs at `http://localhost:8000/docs`

### Accessing the Application

1. **Web Interface**: http://localhost:8000
2. **API Documentation**: http://localhost:8000/docs
3. **Alternative API Docs**: http://localhost:8000/redoc

---

## 📊 Basic Data Analysis Workflow

### Step 1: Upload Your Data
1. Drag and drop your CSV file onto the upload area
2. Or click "Choose CSV File" to select manually
3. Wait for data preview to load

### Step 2: Explore Auto-Generated Suggestions
After upload, you'll see smart suggestions based on your data:
- Dataset overview and summary
- Missing value analysis
- Distribution visualizations
- Correlation analysis
- Time series analysis (if applicable)

### Step 3: Ask Questions
Type natural language questions about your data:

**Basic Queries:**
- "Show me the distribution of column_name"
- "Create a scatter plot of price vs sales"
- "What are the correlations between numeric columns?"
- "Analyze missing values in this dataset"

---

## 🧪 Advanced Statistical Analysis Suite

### Overview
PyData Assistant includes a comprehensive statistical analysis suite with 11 professional-grade tools:

1. **Normality Testing** - 4 methods
2. **Hypothesis Testing** - T-tests, ANOVA, Chi-square
3. **Correlation Analysis** - 3 methods
4. **Outlier Detection** - 4 methods
5. **Regression Analysis** - 3 types
6. **Distribution Fitting** - 13 distributions
7. **Time Series Analysis** - Stationarity and causality tests
8. **Summary Statistics** - Comprehensive metrics

---

## 📈 Statistical Analysis Features

### 1. Normality Testing

**When to Use**: Before applying parametric statistical tests

**Example Prompts**:
```
Test if sales_amount follows a normal distribution
Check normality of customer_age using all available tests
Is the distribution of prices normally distributed?
```

**What You Get**:
- Shapiro-Wilk Test (p-value, statistic)
- D'Agostino-Pearson Test (p-value, statistic)
- Anderson-Darling Test (statistic, critical values)
- Kolmogorov-Smirnov Test (p-value, statistic)

**Interpretation**:
- p-value > 0.05: Data likely follows normal distribution
- p-value ≤ 0.05: Data does NOT follow normal distribution

---

### 2. T-Tests

**When to Use**: Compare means between two groups

**Example Prompts**:
```
Compare average sales between group A and group B using t-test
Perform independent t-test between treatment and control groups
Test if before and after measurements differ significantly (paired)
```

**What You Get**:
- T-statistic
- P-value
- Cohen's d (effect size)
- Confidence interval
- Interpretation

**Effect Size (Cohen's d)**:
- Small: 0.2
- Medium: 0.5
- Large: 0.8

---

### 3. ANOVA (Analysis of Variance)

**When to Use**: Compare means across 3+ groups

**Example Prompts**:
```
Compare sales across different regions using ANOVA
Perform one-way ANOVA on income grouped by education level
Test if product performance differs across categories
```

**What You Get**:
- F-statistic
- P-value
- Eta-squared (effect size)
- Between/within group statistics
- Interpretation

---

### 4. Chi-Square Test

**When to Use**: Test independence between categorical variables

**Example Prompts**:
```
Chi-square test between gender and purchase category
Test independence of region and product preference
Is there a relationship between age_group and subscription_status?
```

**What You Get**:
- Chi-square statistic
- P-value
- Cramér's V (effect size)
- Degrees of freedom
- Contingency table

---

### 5. Correlation Analysis

**When to Use**: Measure strength and direction of relationships

**Example Prompts**:
```
Test correlation between price and sales using Pearson
Analyze relationship between age and income (all methods)
Is there significant correlation between advertising spend and revenue?
```

**Methods Available**:
1. **Pearson**: Linear relationships (requires normal distribution)
2. **Spearman**: Monotonic relationships (rank-based, non-parametric)
3. **Kendall**: Ordinal data (robust to outliers)

**What You Get**:
- Correlation coefficient (-1 to +1)
- P-value (significance)
- Confidence interval

**Interpretation**:
- |r| = 0.0-0.3: Weak
- |r| = 0.3-0.7: Moderate
- |r| = 0.7-1.0: Strong

---

### 6. Outlier Detection

**When to Use**: Identify anomalous data points

**Example Prompts**:
```
Detect outliers in sales_amount using all methods
Find anomalies in customer_spending using Isolation Forest
Identify outliers in transaction_value using IQR method
```

**Methods Available**:
1. **IQR (Interquartile Range)**: Q1 - 1.5×IQR and Q3 + 1.5×IQR
2. **Z-Score**: Values beyond ±3 standard deviations
3. **Modified Z-Score (MAD)**: Robust to outliers, uses median absolute deviation
4. **Isolation Forest**: Machine learning based, detects complex anomalies

**What You Get**:
- Outlier indices
- Outlier values
- Number and percentage of outliers
- Threshold values (method-specific)

---

### 7. Linear Regression

**When to Use**: Model linear relationship between variables

**Example Prompts**:
```
Perform linear regression with advertising as predictor and sales as target
Model relationship between years_experience and salary
Predict revenue based on marketing_spend
```

**What You Get**:
- Coefficients (slope, intercept)
- R² score (goodness of fit)
- Adjusted R²
- RMSE (root mean squared error)
- Residual analysis
- Predictions for test data

**Interpretation**:
- R² = 1.0: Perfect fit
- R² = 0.0: No linear relationship
- Higher R² = Better model fit

---

### 8. Polynomial Regression

**When to Use**: Model non-linear relationships

**Example Prompts**:
```
Fit polynomial regression (degree 2) for temperature vs ice_cream_sales
Model non-linear relationship between age and income with polynomial
Perform cubic regression between time and growth_rate
```

**What You Get**:
- Polynomial coefficients
- R² score
- RMSE
- Predictions
- Degree of polynomial

---

### 9. Logistic Regression

**When to Use**: Predict binary outcomes (0/1, Yes/No)

**Example Prompts**:
```
Logistic regression for churn prediction based on usage metrics
Predict loan approval (approved/rejected) using credit score
Model probability of purchase based on customer features
```

**What You Get**:
- Coefficients
- Accuracy score
- Classification report (precision, recall, F1)
- Confusion matrix
- Predicted probabilities

---

### 10. Distribution Fitting

**When to Use**: Find which probability distribution best fits your data

**Example Prompts**:
```
Fit distributions to customer_age and find best fit
Compare normal and exponential distributions for wait_times
Test if sales_data follows gamma or lognormal distribution
```

**13 Distributions Available**:
- Normal, Exponential, Gamma, Beta
- Lognormal, Weibull, Uniform, Chi-square
- T-distribution, F-distribution, Pareto
- Logistic, Gumbel

**What You Get**:
- Best-fit distribution
- Parameters for each distribution
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Log-likelihood
- Kolmogorov-Smirnov test results
- Good fit indicator

---

### 11. Time Series Analysis

#### A. Stationarity Test

**When to Use**: Before time series modeling (ARIMA, etc.)

**Example Prompts**:
```
Test if daily_sales time series is stationary
Check stationarity of stock_prices using ADF test
Is the temperature time series stationary?
```

**What You Get**:
- ADF statistic
- P-value
- Critical values (1%, 5%, 10%)
- Stationarity conclusion

**Interpretation**:
- p-value < 0.05: Series is stationary
- p-value ≥ 0.05: Series is non-stationary (needs differencing)

#### B. Granger Causality

**When to Use**: Test if one time series predicts another

**Example Prompts**:
```
Test if advertising_spend Granger-causes sales_revenue
Does temperature Granger-cause ice_cream_sales?
Check if social_media_engagement predicts website_traffic
```

**What You Get**:
- F-statistic
- P-values for different lags
- Optimal lag
- Causality conclusion

---

### 12. Summary Statistics

**When to Use**: Get comprehensive descriptive statistics

**Example Prompts**:
```
Get detailed statistics for customer_age
Show comprehensive summary of sales_amount
Analyze distribution characteristics of prices
```

**What You Get**:
- Central tendency: mean, median, mode
- Dispersion: std, variance, range, IQR
- Shape: skewness, kurtosis
- Position: min, max, quartiles, percentiles
- Count and missing values

---

## 💡 Best Practices

### Statistical Analysis Workflow

1. **Upload and Explore**
   - Start with dataset overview
   - Check for missing values
   - Identify data types

2. **Preliminary Analysis**
   - Get summary statistics
   - Visualize distributions
   - Check for outliers

3. **Test Assumptions**
   - Test normality before parametric tests
   - Check for stationarity (time series)
   - Verify homogeneity of variance

4. **Perform Analysis**
   - Choose appropriate test
   - Use specific column names
   - Consider sample size

5. **Interpret Results**
   - Check p-values AND effect sizes
   - Don't rely solely on statistical significance
   - Consider practical significance

6. **Validate Findings**
   - Cross-check with visualizations
   - Test with different methods
   - Document assumptions

---

## 🎯 Example Use Cases

### Business Analytics
```
Test if marketing campaign increased sales (t-test)
Compare customer satisfaction across regions (ANOVA)
Predict customer churn (logistic regression)
Detect fraudulent transactions (outlier detection)
```

### Scientific Research
```
Test normality of experimental measurements
Compare treatment effects (t-test/ANOVA)
Analyze correlation between variables
Fit theoretical distributions to empirical data
```

### Financial Analysis
```
Detect anomalous trading patterns (Isolation Forest)
Test stationarity of stock prices
Model risk-return relationships (regression)
Predict default probability (logistic regression)
```

### Healthcare Analytics
```
Compare treatment outcomes (t-test/ANOVA)
Predict disease diagnosis (logistic regression)
Identify outlier patient metrics
Test if vitals follow normal distribution
```

---

## 🔧 Troubleshooting

### Server Won't Start

**Issue**: `ModuleNotFoundError: No module named 'uvicorn'`

**Solution**:
```bash
pip install -r requirements.txt
```

**Issue**: Port 8000 already in use

**Solution**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Analysis Errors

**Issue**: "Column not found"

**Solution**: Check exact column name (case-sensitive) in Data Preview tab

**Issue**: "Insufficient data"

**Solution**: Ensure enough rows for statistical test (typically need 30+)

**Issue**: "Non-numeric column"

**Solution**: Use numeric columns for statistical tests, categorical for chi-square

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Statistical Documentation**: `STATISTICAL_ANALYSIS_DOCUMENTATION.md`
- **Quick Start Guide**: `STATISTICAL_FEATURES_QUICKSTART.md`
- **Test Examples**: `test_statistical_suite.py`

---

## 🆘 Getting Help

1. Click the **Help** button (?) in the header
2. View **Examples** after uploading data
3. Check **API docs** at `/docs`
4. Review test file for code examples

---

## 📝 Tips for Better Results

1. **Use exact column names** - Copy from Data Preview
2. **Check data types** - Numeric for most statistical tests
3. **Handle missing data** - Address before analysis
4. **Start simple** - Begin with summary statistics
5. **Verify assumptions** - Test normality first
6. **Interpret carefully** - Consider both statistical and practical significance
7. **Document workflow** - Export results for reproducibility

---

**Version**: 2.0  
**Last Updated**: October 2025  
**Features**: 11 Statistical Analysis Tools + AI-Powered Insights
