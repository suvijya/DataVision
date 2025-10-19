"""
Advanced Statistical Analysis Suite
Provides comprehensive statistical testing, regression analysis, time series forecasting,
outlier detection, distribution fitting, and Bayesian statistics.
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    normaltest, shapiro, anderson, kstest,
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    chi2_contingency, f_oneway, kruskal,
    pearsonr, spearmanr, kendalltau,
    skew, kurtosis, jarque_bera
)
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Standardized result for statistical tests."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str
    confidence_level: float = 0.95
    additional_info: Dict[str, Any] = None


@dataclass
class RegressionResult:
    """Result for regression analysis."""
    model_type: str
    r_squared: float
    adjusted_r_squared: Optional[float]
    coefficients: List[float]
    intercept: float
    predictions: List[float]
    residuals: List[float]
    rmse: float
    mae: float
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class OutlierDetectionResult:
    """Result for outlier detection."""
    method: str
    outlier_indices: List[int]
    outlier_values: List[float]
    outlier_count: int
    outlier_percentage: float
    bounds: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None


class AdvancedStatisticalAnalysis:
    """
    Comprehensive statistical analysis toolkit for professional data analysis.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Args:
            df: Pandas DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # ============================================================================
    # HYPOTHESIS TESTING
    # ============================================================================
    
    def test_normality(
        self, 
        column: str, 
        alpha: float = 0.05,
        methods: List[str] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        Comprehensive normality testing using multiple methods.
        
        Args:
            column: Column name to test
            alpha: Significance level (default 0.05)
            methods: List of methods to use. Options: ['shapiro', 'dagostino', 'anderson', 'ks']
                    If None, uses all applicable methods
        
        Returns:
            Dictionary of test results
        """
        if column not in self.numeric_cols:
            raise ValueError(f"Column '{column}' is not numeric")
        
        data = self.df[column].dropna()
        
        if len(data) < 3:
            raise ValueError("Need at least 3 data points for normality tests")
        
        results = {}
        all_methods = methods or ['shapiro', 'dagostino', 'anderson', 'ks']
        
        # Shapiro-Wilk Test (best for n < 5000)
        if 'shapiro' in all_methods and len(data) <= 5000:
            try:
                stat, p_value = shapiro(data)
                results['shapiro_wilk'] = StatisticalTestResult(
                    test_name="Shapiro-Wilk Test",
                    statistic=float(stat),
                    p_value=float(p_value),
                    significant=p_value < alpha,
                    interpretation=f"Data is {'NOT ' if p_value < alpha else ''}normally distributed (p={p_value:.4f})",
                    confidence_level=1 - alpha,
                    additional_info={
                        'sample_size': len(data),
                        'recommended_for': 'n < 5000'
                    }
                )
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed: {e}")
        
        # D'Agostino-Pearson Test
        if 'dagostino' in all_methods and len(data) >= 20:
            try:
                stat, p_value = normaltest(data)
                results['dagostino_pearson'] = StatisticalTestResult(
                    test_name="D'Agostino-Pearson Test",
                    statistic=float(stat),
                    p_value=float(p_value),
                    significant=p_value < alpha,
                    interpretation=f"Data is {'NOT ' if p_value < alpha else ''}normally distributed (p={p_value:.4f})",
                    confidence_level=1 - alpha,
                    additional_info={
                        'sample_size': len(data),
                        'skewness': float(skew(data)),
                        'kurtosis': float(kurtosis(data))
                    }
                )
            except Exception as e:
                logger.warning(f"D'Agostino-Pearson test failed: {e}")
        
        # Anderson-Darling Test
        if 'anderson' in all_methods:
            try:
                result = anderson(data)
                # Use 5% significance level (index 2)
                critical_value = result.critical_values[2]
                significant = result.statistic > critical_value
                
                results['anderson_darling'] = StatisticalTestResult(
                    test_name="Anderson-Darling Test",
                    statistic=float(result.statistic),
                    p_value=0.05,  # Anderson test doesn't provide exact p-value
                    significant=significant,
                    interpretation=f"Data is {'NOT ' if significant else ''}normally distributed (statistic={result.statistic:.4f}, critical={critical_value:.4f})",
                    confidence_level=0.95,
                    additional_info={
                        'critical_values': result.critical_values.tolist(),
                        'significance_levels': result.significance_level.tolist()
                    }
                )
            except Exception as e:
                logger.warning(f"Anderson-Darling test failed: {e}")
        
        # Kolmogorov-Smirnov Test
        if 'ks' in all_methods:
            try:
                # Test against normal distribution with same mean and std
                mean, std = data.mean(), data.std()
                stat, p_value = kstest(data, lambda x: stats.norm.cdf(x, mean, std))
                
                results['kolmogorov_smirnov'] = StatisticalTestResult(
                    test_name="Kolmogorov-Smirnov Test",
                    statistic=float(stat),
                    p_value=float(p_value),
                    significant=p_value < alpha,
                    interpretation=f"Data is {'NOT ' if p_value < alpha else ''}normally distributed (p={p_value:.4f})",
                    confidence_level=1 - alpha,
                    additional_info={
                        'mean': float(mean),
                        'std': float(std)
                    }
                )
            except Exception as e:
                logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
        
        return results
    
    def t_test_independent(
        self, 
        group_col: str, 
        value_col: str,
        group1_value: Any,
        group2_value: Any,
        alpha: float = 0.05,
        equal_var: bool = True
    ) -> StatisticalTestResult:
        """
        Independent samples t-test.
        
        Args:
            group_col: Column containing group labels
            value_col: Column containing numeric values
            group1_value: Value identifying first group
            group2_value: Value identifying second group
            alpha: Significance level
            equal_var: Assume equal variances (True) or Welch's t-test (False)
        
        Returns:
            Statistical test result
        """
        group1 = self.df[self.df[group_col] == group1_value][value_col].dropna()
        group2 = self.df[self.df[group_col] == group2_value][value_col].dropna()
        
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Each group must have at least 2 observations")
        
        stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)
        
        mean_diff = group1.mean() - group2.mean()
        pooled_std = np.sqrt(((len(group1) - 1) * group1.std()**2 + (len(group2) - 1) * group2.std()**2) / (len(group1) + len(group2) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return StatisticalTestResult(
            test_name="Independent Samples T-Test" + (" (Welch's)" if not equal_var else ""),
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < alpha,
            interpretation=f"Groups are {'significantly' if p_value < alpha else 'not significantly'} different (p={p_value:.4f})",
            confidence_level=1 - alpha,
            additional_info={
                'group1_mean': float(group1.mean()),
                'group2_mean': float(group2.mean()),
                'group1_std': float(group1.std()),
                'group2_std': float(group2.std()),
                'group1_n': len(group1),
                'group2_n': len(group2),
                'mean_difference': float(mean_diff),
                'cohens_d': float(cohens_d),
                'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
            }
        )
    
    def t_test_paired(
        self,
        before_col: str,
        after_col: str,
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Paired samples t-test.
        
        Args:
            before_col: Column with before measurements
            after_col: Column with after measurements
            alpha: Significance level
        
        Returns:
            Statistical test result
        """
        # Get paired data (drop rows with missing values in either column)
        data = self.df[[before_col, after_col]].dropna()
        before = data[before_col]
        after = data[after_col]
        
        if len(before) < 2:
            raise ValueError("Need at least 2 paired observations")
        
        stat, p_value = ttest_rel(before, after)
        
        mean_diff = (after - before).mean()
        std_diff = (after - before).std()
        
        return StatisticalTestResult(
            test_name="Paired Samples T-Test",
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < alpha,
            interpretation=f"Paired measurements are {'significantly' if p_value < alpha else 'not significantly'} different (p={p_value:.4f})",
            confidence_level=1 - alpha,
            additional_info={
                'before_mean': float(before.mean()),
                'after_mean': float(after.mean()),
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                'n_pairs': len(before)
            }
        )
    
    def anova_one_way(
        self,
        group_col: str,
        value_col: str,
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        One-way ANOVA test.
        
        Args:
            group_col: Column containing group labels
            value_col: Column containing numeric values
            alpha: Significance level
        
        Returns:
            Statistical test result
        """
        groups = []
        group_names = []
        
        for name, group in self.df.groupby(group_col):
            data = group[value_col].dropna()
            if len(data) > 0:
                groups.append(data)
                group_names.append(name)
        
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for ANOVA")
        
        stat, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        grand_mean = pd.concat(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum(sum((x - grand_mean)**2 for x in g) for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return StatisticalTestResult(
            test_name="One-Way ANOVA",
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < alpha,
            interpretation=f"Groups are {'significantly' if p_value < alpha else 'not significantly'} different (p={p_value:.4f})",
            confidence_level=1 - alpha,
            additional_info={
                'n_groups': len(groups),
                'group_names': [str(name) for name in group_names],
                'group_means': [float(g.mean()) for g in groups],
                'group_sizes': [len(g) for g in groups],
                'eta_squared': float(eta_squared),
                'effect_size': 'small' if eta_squared < 0.06 else ('medium' if eta_squared < 0.14 else 'large')
            }
        )
    
    def chi_square_test(
        self,
        col1: str,
        col2: str,
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Chi-square test of independence.
        
        Args:
            col1: First categorical column
            col2: Second categorical column
            alpha: Significance level
        
        Returns:
            Statistical test result
        """
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V for effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(stat / (n * min_dim)) if min_dim > 0 else 0
        
        return StatisticalTestResult(
            test_name="Chi-Square Test of Independence",
            statistic=float(stat),
            p_value=float(p_value),
            significant=p_value < alpha,
            interpretation=f"Variables are {'dependent' if p_value < alpha else 'independent'} (p={p_value:.4f})",
            confidence_level=1 - alpha,
            additional_info={
                'degrees_of_freedom': int(dof),
                'contingency_table': contingency_table.to_dict(),
                'cramers_v': float(cramers_v),
                'effect_size': 'small' if cramers_v < 0.3 else ('medium' if cramers_v < 0.5 else 'large')
            }
        )
    
    def correlation_test(
        self,
        col1: str,
        col2: str,
        method: str = 'pearson',
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        Test correlation significance.
        
        Args:
            col1: First numeric column
            col2: Second numeric column
            method: 'pearson', 'spearman', or 'kendall'
            alpha: Significance level
        
        Returns:
            Statistical test result
        """
        data = self.df[[col1, col2]].dropna()
        x, y = data[col1], data[col2]
        
        if len(x) < 3:
            raise ValueError("Need at least 3 observations for correlation test")
        
        if method == 'pearson':
            corr, p_value = pearsonr(x, y)
            test_name = "Pearson Correlation"
        elif method == 'spearman':
            corr, p_value = spearmanr(x, y)
            test_name = "Spearman Rank Correlation"
        elif method == 'kendall':
            corr, p_value = kendalltau(x, y)
            test_name = "Kendall Tau Correlation"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return StatisticalTestResult(
            test_name=test_name,
            statistic=float(corr),
            p_value=float(p_value),
            significant=p_value < alpha,
            interpretation=f"Correlation is {'significant' if p_value < alpha else 'not significant'} (r={corr:.4f}, p={p_value:.4f})",
            confidence_level=1 - alpha,
            additional_info={
                'correlation': float(corr),
                'n_observations': len(x),
                'strength': 'weak' if abs(corr) < 0.3 else ('moderate' if abs(corr) < 0.7 else 'strong'),
                'direction': 'positive' if corr > 0 else 'negative'
            }
        )
    
    # ============================================================================
    # OUTLIER DETECTION
    # ============================================================================
    
    def detect_outliers_iqr(
        self,
        column: str,
        multiplier: float = 1.5
    ) -> OutlierDetectionResult:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            column: Column name
            multiplier: IQR multiplier (typically 1.5 or 3.0)
        
        Returns:
            Outlier detection result
        """
        data = self.df[column].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers_mask = (data < lower_bound) | (data > upper_bound)
        outlier_indices = data[outliers_mask].index.tolist()
        outlier_values = data[outliers_mask].tolist()
        
        return OutlierDetectionResult(
            method="IQR Method",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_count=len(outlier_values),
            outlier_percentage=len(outlier_values) / len(data) * 100,
            bounds={'lower': float(lower_bound), 'upper': float(upper_bound)},
            threshold=multiplier
        )
    
    def detect_outliers_zscore(
        self,
        column: str,
        threshold: float = 3.0
    ) -> OutlierDetectionResult:
        """
        Detect outliers using Z-score method.
        
        Args:
            column: Column name
            threshold: Z-score threshold (typically 3.0)
        
        Returns:
            Outlier detection result
        """
        data = self.df[column].dropna()
        z_scores = np.abs(stats.zscore(data))
        
        outliers_mask = z_scores > threshold
        outlier_indices = data[outliers_mask].index.tolist()
        outlier_values = data[outliers_mask].tolist()
        
        return OutlierDetectionResult(
            method="Z-Score Method",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_count=len(outlier_values),
            outlier_percentage=len(outlier_values) / len(data) * 100,
            threshold=threshold
        )
    
    def detect_outliers_modified_zscore(
        self,
        column: str,
        threshold: float = 3.5
    ) -> OutlierDetectionResult:
        """
        Detect outliers using Modified Z-score (MAD - Median Absolute Deviation).
        More robust to outliers than standard Z-score.
        
        Args:
            column: Column name
            threshold: Modified Z-score threshold (typically 3.5)
        
        Returns:
            Outlier detection result
        """
        data = self.df[column].dropna()
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            # If MAD is 0, all values are the same
            return OutlierDetectionResult(
                method="Modified Z-Score (MAD)",
                outlier_indices=[],
                outlier_values=[],
                outlier_count=0,
                outlier_percentage=0.0,
                threshold=threshold
            )
        
        modified_z_scores = 0.6745 * (data - median) / mad
        
        outliers_mask = np.abs(modified_z_scores) > threshold
        outlier_indices = data[outliers_mask].index.tolist()
        outlier_values = data[outliers_mask].tolist()
        
        return OutlierDetectionResult(
            method="Modified Z-Score (MAD)",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_count=len(outlier_values),
            outlier_percentage=len(outlier_values) / len(data) * 100,
            threshold=threshold
        )
    
    def detect_outliers_isolation_forest(
        self,
        column: str,
        contamination: float = 0.1
    ) -> OutlierDetectionResult:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Args:
            column: Column name
            contamination: Expected proportion of outliers
        
        Returns:
            Outlier detection result
        """
        data = self.df[column].dropna()
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(data.values.reshape(-1, 1))
        
        outliers_mask = predictions == -1
        outlier_indices = data[outliers_mask].index.tolist()
        outlier_values = data[outliers_mask].tolist()
        
        return OutlierDetectionResult(
            method="Isolation Forest",
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_count=len(outlier_values),
            outlier_percentage=len(outlier_values) / len(data) * 100,
            threshold=contamination
        )
    
    def detect_outliers_all_methods(
        self,
        column: str
    ) -> Dict[str, OutlierDetectionResult]:
        """
        Apply all outlier detection methods and return comprehensive results.
        
        Args:
            column: Column name
        
        Returns:
            Dictionary of results from all methods
        """
        results = {}
        
        try:
            results['iqr'] = self.detect_outliers_iqr(column)
        except Exception as e:
            logger.warning(f"IQR method failed: {e}")
        
        try:
            results['zscore'] = self.detect_outliers_zscore(column)
        except Exception as e:
            logger.warning(f"Z-score method failed: {e}")
        
        try:
            results['modified_zscore'] = self.detect_outliers_modified_zscore(column)
        except Exception as e:
            logger.warning(f"Modified Z-score method failed: {e}")
        
        try:
            results['isolation_forest'] = self.detect_outliers_isolation_forest(column)
        except Exception as e:
            logger.warning(f"Isolation Forest method failed: {e}")
        
        return results
    
    # ============================================================================
    # REGRESSION ANALYSIS
    # ============================================================================
    
    def linear_regression(
        self,
        x_col: Union[str, List[str]],
        y_col: str
    ) -> RegressionResult:
        """
        Perform linear regression analysis.
        
        Args:
            x_col: Independent variable(s) - single column name or list
            y_col: Dependent variable
        
        Returns:
            Regression result
        """
        # Prepare data
        if isinstance(x_col, str):
            x_cols = [x_col]
        else:
            x_cols = x_col
        
        data = self.df[x_cols + [y_col]].dropna()
        X = data[x_cols].values.reshape(-1, len(x_cols))
        y = data[y_col].values
        
        if len(X) < len(x_cols) + 2:
            raise ValueError("Need more observations than variables")
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions and metrics
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = len(x_cols)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Feature importance (standardized coefficients)
        feature_importance = {}
        if len(x_cols) > 0:
            X_std = (X - X.mean(axis=0)) / X.std(axis=0)
            y_std = (y - y.mean()) / y.std()
            std_coef = np.dot(X_std.T, y_std) / len(y_std)
            
            for i, col in enumerate(x_cols):
                feature_importance[col] = float(abs(std_coef[i]))
        
        return RegressionResult(
            model_type="Linear Regression",
            r_squared=float(r2),
            adjusted_r_squared=float(adj_r2) if adj_r2 is not None else None,
            coefficients=[float(c) for c in model.coef_],
            intercept=float(model.intercept_),
            predictions=y_pred.tolist(),
            residuals=residuals.tolist(),
            rmse=float(rmse),
            mae=float(mae),
            feature_importance=feature_importance
        )
    
    def polynomial_regression(
        self,
        x_col: str,
        y_col: str,
        degree: int = 2
    ) -> RegressionResult:
        """
        Perform polynomial regression analysis.
        
        Args:
            x_col: Independent variable
            y_col: Dependent variable
            degree: Polynomial degree
        
        Returns:
            Regression result
        """
        data = self.df[[x_col, y_col]].dropna()
        X = data[x_col].values.reshape(-1, 1)
        y = data[y_col].values
        
        # Transform features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predictions and metrics
        y_pred = model.predict(X_poly)
        residuals = y - y_pred
        
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = degree
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else None
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        return RegressionResult(
            model_type=f"Polynomial Regression (degree={degree})",
            r_squared=float(r2),
            adjusted_r_squared=float(adj_r2) if adj_r2 is not None else None,
            coefficients=[float(c) for c in model.coef_],
            intercept=float(model.intercept_),
            predictions=y_pred.tolist(),
            residuals=residuals.tolist(),
            rmse=float(rmse),
            mae=float(mae),
            feature_importance={f"{x_col}^{i}": float(abs(c)) for i, c in enumerate(model.coef_)}
        )
    
    def logistic_regression(
        self,
        x_col: Union[str, List[str]],
        y_col: str
    ) -> Dict[str, Any]:
        """
        Perform logistic regression analysis.
        
        Args:
            x_col: Independent variable(s)
            y_col: Binary dependent variable
        
        Returns:
            Dictionary with logistic regression results
        """
        # Prepare data
        if isinstance(x_col, str):
            x_cols = [x_col]
        else:
            x_cols = x_col
        
        data = self.df[x_cols + [y_col]].dropna()
        X = data[x_cols].values.reshape(-1, len(x_cols))
        y = data[y_col].values
        
        # Check if binary
        unique_vals = np.unique(y)
        if len(unique_vals) != 2:
            raise ValueError(f"Target variable must be binary, found {len(unique_vals)} unique values")
        
        # Fit model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # Accuracy
        accuracy = (y == y_pred).sum() / len(y)
        
        # Confusion matrix
        tp = ((y == 1) & (y_pred == 1)).sum()
        tn = ((y == 0) & (y_pred == 0)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'model_type': 'Logistic Regression',
            'coefficients': [float(c) for c in model.coef_[0]],
            'intercept': float(model.intercept_[0]),
            'feature_names': x_cols,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'confusion_matrix': {
                'true_positive': int(tp),
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn)
            },
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist()
        }
    
    # ============================================================================
    # DISTRIBUTION FITTING
    # ============================================================================
    
    def fit_distributions(
        self,
        column: str,
        distributions: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit multiple distributions and find best fit.
        
        Args:
            column: Column name
            distributions: List of distribution names to try
                          Default: ['norm', 'expon', 'gamma', 'lognorm', 'weibull_min']
        
        Returns:
            Dictionary with fit results for each distribution
        """
        data = self.df[column].dropna()
        
        if distributions is None:
            distributions = ['norm', 'expon', 'gamma', 'lognorm', 'weibull_min']
        
        results = {}
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                
                # Fit distribution
                params = dist.fit(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.kstest(data, dist_name, args=params)
                
                # AIC and BIC
                log_likelihood = np.sum(dist.logpdf(data, *params))
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                results[dist_name] = {
                    'distribution': dist_name,
                    'parameters': {f'param_{i}': float(p) for i, p in enumerate(params)},
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(log_likelihood),
                    'good_fit': bool(ks_pvalue > 0.05)
                }
                
            except Exception as e:
                logger.warning(f"Failed to fit {dist_name}: {e}")
                results[dist_name] = {
                    'distribution': dist_name,
                    'error': str(e),
                    'good_fit': False
                }
        
        # Find best fit based on AIC
        valid_results = {k: v for k, v in results.items() if 'aic' in v}
        if valid_results:
            best_dist = min(valid_results.items(), key=lambda x: x[1]['aic'])
            for result in results.values():
                result['is_best_fit'] = False
            results[best_dist[0]]['is_best_fit'] = True
        
        return results
    
    # ============================================================================
    # TIME SERIES ANALYSIS
    # ============================================================================
    
    def test_stationarity(
        self,
        column: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test time series stationarity using Augmented Dickey-Fuller test.
        
        Args:
            column: Column name
            alpha: Significance level
        
        Returns:
            Dictionary with stationarity test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        data = self.df[column].dropna()
        
        result = adfuller(data, autolag='AIC')
        
        return {
            'test_name': 'Augmented Dickey-Fuller Test',
            'adf_statistic': float(result[0]),
            'p_value': float(result[1]),
            'used_lag': int(result[2]),
            'n_observations': int(result[3]),
            'critical_values': {k: float(v) for k, v in result[4].items()},
            'is_stationary': result[1] < alpha,
            'interpretation': f"Series is {'stationary' if result[1] < alpha else 'non-stationary'} (p={result[1]:.4f})"
        }
    
    def granger_causality(
        self,
        cause_col: str,
        effect_col: str,
        max_lag: int = 5,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test Granger causality between two time series.
        
        Args:
            cause_col: Potential cause variable
            effect_col: Effect variable
            max_lag: Maximum lag to test
            alpha: Significance level
        
        Returns:
            Dictionary with Granger causality results
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        data = self.df[[effect_col, cause_col]].dropna()
        
        if len(data) < max_lag + 10:
            raise ValueError(f"Need at least {max_lag + 10} observations")
        
        # Run Granger causality tests
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Extract results for each lag
        lag_results = {}
        significant_lags = []
        
        for lag in range(1, max_lag + 1):
            # Get F-test p-value
            f_test = results[lag][0]['ssr_ftest']
            p_value = f_test[1]
            
            lag_results[f'lag_{lag}'] = {
                'f_statistic': float(f_test[0]),
                'p_value': float(p_value),
                'significant': p_value < alpha
            }
            
            if p_value < alpha:
                significant_lags.append(lag)
        
        return {
            'test_name': 'Granger Causality Test',
            'cause_variable': cause_col,
            'effect_variable': effect_col,
            'max_lag_tested': max_lag,
            'lag_results': lag_results,
            'significant_lags': significant_lags,
            'granger_causes': len(significant_lags) > 0,
            'interpretation': f"'{cause_col}' {'does' if len(significant_lags) > 0 else 'does not'} Granger-cause '{effect_col}' at {int((1-alpha)*100)}% confidence"
        }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def get_summary_statistics(self, column: str) -> Dict[str, float]:
        """Get comprehensive summary statistics for a column."""
        data = self.df[column].dropna()
        
        return {
            'count': int(len(data)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'q1': float(data.quantile(0.25)),
            'median': float(data.median()),
            'q3': float(data.quantile(0.75)),
            'max': float(data.max()),
            'skewness': float(skew(data)),
            'kurtosis': float(kurtosis(data)),
            'range': float(data.max() - data.min()),
            'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
            'variance': float(data.var()),
            'cv': float(data.std() / data.mean()) if data.mean() != 0 else None
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def convert_results_to_dict(obj: Any) -> Any:
    """Convert dataclass results to dictionaries for JSON serialization."""
    if isinstance(obj, (StatisticalTestResult, RegressionResult, OutlierDetectionResult)):
        result = {}
        for field_name, field_value in obj.__dict__.items():
            result[field_name] = convert_results_to_dict(field_value)
        return result
    elif isinstance(obj, dict):
        return {k: convert_results_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_results_to_dict(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None:
        return None
    else:
        return obj

