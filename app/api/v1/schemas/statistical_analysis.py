"""
Pydantic schemas for advanced statistical analysis endpoints.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class NormalityTestRequest(BaseModel):
    """Request for normality testing."""
    session_id: str = Field(..., description="Session ID")
    column: str = Field(..., description="Column name to test")
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)
    methods: Optional[List[str]] = Field(None, description="List of methods: shapiro, dagostino, anderson, ks")


class TTestRequest(BaseModel):
    """Request for t-test."""
    session_id: str = Field(..., description="Session ID")
    test_type: str = Field(..., description="Type of t-test: independent or paired")
    group_col: Optional[str] = Field(None, description="Column containing groups (for independent)")
    value_col: Optional[str] = Field(None, description="Column containing values (for independent)")
    group1_value: Optional[Any] = Field(None, description="Value for group 1 (for independent)")
    group2_value: Optional[Any] = Field(None, description="Value for group 2 (for independent)")
    before_col: Optional[str] = Field(None, description="Before measurements column (for paired)")
    after_col: Optional[str] = Field(None, description="After measurements column (for paired)")
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)
    equal_var: bool = Field(True, description="Assume equal variances")


class ANOVARequest(BaseModel):
    """Request for ANOVA test."""
    session_id: str = Field(..., description="Session ID")
    group_col: str = Field(..., description="Column containing groups")
    value_col: str = Field(..., description="Column containing values")
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)


class ChiSquareRequest(BaseModel):
    """Request for chi-square test."""
    session_id: str = Field(..., description="Session ID")
    col1: str = Field(..., description="First categorical column")
    col2: str = Field(..., description="Second categorical column")
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)


class CorrelationTestRequest(BaseModel):
    """Request for correlation test."""
    session_id: str = Field(..., description="Session ID")
    col1: str = Field(..., description="First numeric column")
    col2: str = Field(..., description="Second numeric column")
    method: str = Field("pearson", description="Method: pearson, spearman, or kendall")
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)


class OutlierDetectionRequest(BaseModel):
    """Request for outlier detection."""
    session_id: str = Field(..., description="Session ID")
    column: str = Field(..., description="Column name")
    method: str = Field("all", description="Method: iqr, zscore, modified_zscore, isolation_forest, or all")
    iqr_multiplier: float = Field(1.5, description="IQR multiplier", gt=0)
    zscore_threshold: float = Field(3.0, description="Z-score threshold", gt=0)
    modified_zscore_threshold: float = Field(3.5, description="Modified Z-score threshold", gt=0)
    isolation_contamination: float = Field(0.1, description="Isolation Forest contamination", ge=0.01, le=0.5)


class RegressionRequest(BaseModel):
    """Request for regression analysis."""
    session_id: str = Field(..., description="Session ID")
    regression_type: str = Field(..., description="Type: linear, polynomial, or logistic")
    x_col: Union[str, List[str]] = Field(..., description="Independent variable(s)")
    y_col: str = Field(..., description="Dependent variable")
    polynomial_degree: Optional[int] = Field(2, description="Degree for polynomial regression", ge=2, le=10)


class DistributionFitRequest(BaseModel):
    """Request for distribution fitting."""
    session_id: str = Field(..., description="Session ID")
    column: str = Field(..., description="Column name")
    distributions: Optional[List[str]] = Field(
        None, 
        description="List of distributions to try: norm, expon, gamma, lognorm, weibull_min"
    )


class StationarityTestRequest(BaseModel):
    """Request for time series stationarity test."""
    session_id: str = Field(..., description="Session ID")
    column: str = Field(..., description="Column name")
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)


class GrangerCausalityRequest(BaseModel):
    """Request for Granger causality test."""
    session_id: str = Field(..., description="Session ID")
    cause_col: str = Field(..., description="Potential cause variable")
    effect_col: str = Field(..., description="Effect variable")
    max_lag: int = Field(5, description="Maximum lag to test", ge=1, le=20)
    alpha: float = Field(0.05, description="Significance level", ge=0.01, le=0.1)


class SummaryStatisticsRequest(BaseModel):
    """Request for summary statistics."""
    session_id: str = Field(..., description="Session ID")
    column: str = Field(..., description="Column name")


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class StatisticalTestResponse(BaseModel):
    """Response for statistical tests."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    test_results: Dict[str, Any]
    message: str
    execution_time: float


class OutlierDetectionResponse(BaseModel):
    """Response for outlier detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    outlier_results: Dict[str, Any]
    message: str
    execution_time: float


class RegressionResponse(BaseModel):
    """Response for regression analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    regression_results: Dict[str, Any]
    message: str
    execution_time: float


class DistributionFitResponse(BaseModel):
    """Response for distribution fitting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    fit_results: Dict[str, Any]
    best_distribution: Optional[str]
    message: str
    execution_time: float


class TimeSeriesAnalysisResponse(BaseModel):
    """Response for time series analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    analysis_results: Dict[str, Any]
    message: str
    execution_time: float


class SummaryStatisticsResponse(BaseModel):
    """Response for summary statistics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    statistics: Dict[str, Any]
    message: str
    execution_time: float


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    status_code: int
