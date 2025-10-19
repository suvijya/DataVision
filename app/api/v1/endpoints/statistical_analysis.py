"""
FastAPI router for advanced statistical analysis endpoints.
Provides hypothesis testing, regression, outlier detection, and more.
"""

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services.session_manager import get_session, Session
from app.services.statistical_analysis import (
    AdvancedStatisticalAnalysis,
    convert_results_to_dict
)
from app.api.v1.schemas.statistical_analysis import (
    NormalityTestRequest,
    TTestRequest,
    ANOVARequest,
    ChiSquareRequest,
    CorrelationTestRequest,
    OutlierDetectionRequest,
    RegressionRequest,
    DistributionFitRequest,
    StationarityTestRequest,
    GrangerCausalityRequest,
    SummaryStatisticsRequest,
    StatisticalTestResponse,
    OutlierDetectionResponse,
    RegressionResponse,
    DistributionFitResponse,
    TimeSeriesAnalysisResponse,
    SummaryStatisticsResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session_from_request(session_id: str) -> Session:
    """Get session from request, raising HTTPException if not found."""
    try:
        return get_session(session_id)
    except ValueError as e:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# HYPOTHESIS TESTING ENDPOINTS
# ============================================================================

@router.post(
    "/statistical-analysis/normality-test",
    response_model=StatisticalTestResponse,
    summary="Test data normality",
    description="Perform normality tests using Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, and/or Kolmogorov-Smirnov tests"
)
async def test_normality(request: NormalityTestRequest):
    """Test if data follows a normal distribution."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        results = analyzer.test_normality(
            column=request.column,
            alpha=request.alpha,
            methods=request.methods
        )
        
        results_dict = convert_results_to_dict(results)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "test_results": results_dict,
                "message": f"Normality tests completed for column '{request.column}'",
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in normality test: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Normality test failed",
                "execution_time": time.time() - start_time
            }
        )


@router.post(
    "/statistical-analysis/t-test",
    response_model=StatisticalTestResponse,
    summary="Perform t-test",
    description="Perform independent samples or paired samples t-test"
)
async def perform_t_test(request: TTestRequest):
    """Perform t-test (independent or paired)."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        if request.test_type == "independent":
            if not all([request.group_col, request.value_col, request.group1_value, request.group2_value]):
                raise ValueError("Missing required parameters for independent t-test")
            
            result = analyzer.t_test_independent(
                group_col=request.group_col,
                value_col=request.value_col,
                group1_value=request.group1_value,
                group2_value=request.group2_value,
                alpha=request.alpha,
                equal_var=request.equal_var
            )
            
        elif request.test_type == "paired":
            if not all([request.before_col, request.after_col]):
                raise ValueError("Missing required parameters for paired t-test")
            
            result = analyzer.t_test_paired(
                before_col=request.before_col,
                after_col=request.after_col,
                alpha=request.alpha
            )
        else:
            raise ValueError(f"Invalid test_type: {request.test_type}. Use 'independent' or 'paired'")
        
        result_dict = convert_results_to_dict(result)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "test_results": result_dict,
                "message": f"{request.test_type.capitalize()} t-test completed successfully",
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in t-test: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "T-test failed",
                "execution_time": time.time() - start_time
            }
        )


@router.post(
    "/statistical-analysis/anova",
    response_model=StatisticalTestResponse,
    summary="Perform ANOVA test",
    description="Perform one-way ANOVA to compare means across multiple groups"
)
async def perform_anova(request: ANOVARequest):
    """Perform one-way ANOVA."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        result = analyzer.anova_one_way(
            group_col=request.group_col,
            value_col=request.value_col,
            alpha=request.alpha
        )
        
        result_dict = convert_results_to_dict(result)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "test_results": result_dict,
                "message": "ANOVA test completed successfully",
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in ANOVA: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "ANOVA test failed",
                "execution_time": time.time() - start_time
            }
        )


@router.post(
    "/statistical-analysis/chi-square",
    response_model=StatisticalTestResponse,
    summary="Perform chi-square test",
    description="Test independence between two categorical variables"
)
async def perform_chi_square(request: ChiSquareRequest):
    """Perform chi-square test of independence."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        result = analyzer.chi_square_test(
            col1=request.col1,
            col2=request.col2,
            alpha=request.alpha
        )
        
        result_dict = convert_results_to_dict(result)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "test_results": result_dict,
                "message": "Chi-square test completed successfully",
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chi-square test: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Chi-square test failed",
                "execution_time": time.time() - start_time
            }
        )


@router.post(
    "/statistical-analysis/correlation-test",
    response_model=StatisticalTestResponse,
    summary="Test correlation significance",
    description="Test correlation between two numeric variables using Pearson, Spearman, or Kendall methods"
)
async def test_correlation(request: CorrelationTestRequest):
    """Test correlation significance."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        result = analyzer.correlation_test(
            col1=request.col1,
            col2=request.col2,
            method=request.method,
            alpha=request.alpha
        )
        
        result_dict = convert_results_to_dict(result)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "test_results": result_dict,
                "message": f"{request.method.capitalize()} correlation test completed successfully",
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in correlation test: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Correlation test failed",
                "execution_time": time.time() - start_time
            }
        )


# ============================================================================
# OUTLIER DETECTION ENDPOINTS
# ============================================================================

@router.post(
    "/statistical-analysis/outlier-detection",
    response_model=OutlierDetectionResponse,
    summary="Detect outliers",
    description="Detect outliers using IQR, Z-score, Modified Z-score, or Isolation Forest methods"
)
async def detect_outliers(request: OutlierDetectionRequest):
    """Detect outliers in data."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        if request.method == "all":
            results = analyzer.detect_outliers_all_methods(request.column)
        elif request.method == "iqr":
            results = {
                "iqr": analyzer.detect_outliers_iqr(request.column, request.iqr_multiplier)
            }
        elif request.method == "zscore":
            results = {
                "zscore": analyzer.detect_outliers_zscore(request.column, request.zscore_threshold)
            }
        elif request.method == "modified_zscore":
            results = {
                "modified_zscore": analyzer.detect_outliers_modified_zscore(request.column, request.modified_zscore_threshold)
            }
        elif request.method == "isolation_forest":
            results = {
                "isolation_forest": analyzer.detect_outliers_isolation_forest(request.column, request.isolation_contamination)
            }
        else:
            raise ValueError(f"Invalid method: {request.method}")
        
        results_dict = convert_results_to_dict(results)
        
        return OutlierDetectionResponse(
            success=True,
            outlier_results=results_dict,
            message=f"Outlier detection completed for column '{request.column}' using {request.method} method(s)",
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in outlier detection: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Outlier detection failed",
                "execution_time": time.time() - start_time
            }
        )


# ============================================================================
# REGRESSION ANALYSIS ENDPOINTS
# ============================================================================

@router.post(
    "/statistical-analysis/regression",
    response_model=RegressionResponse,
    summary="Perform regression analysis",
    description="Perform linear, polynomial, or logistic regression"
)
async def perform_regression(request: RegressionRequest):
    """Perform regression analysis."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        if request.regression_type == "linear":
            result = analyzer.linear_regression(
                x_col=request.x_col,
                y_col=request.y_col
            )
            result_dict = convert_results_to_dict(result)
            
        elif request.regression_type == "polynomial":
            if isinstance(request.x_col, list) and len(request.x_col) > 1:
                raise ValueError("Polynomial regression only supports single independent variable")
            
            x_col_name = request.x_col if isinstance(request.x_col, str) else request.x_col[0]
            result = analyzer.polynomial_regression(
                x_col=x_col_name,
                y_col=request.y_col,
                degree=request.polynomial_degree
            )
            result_dict = convert_results_to_dict(result)
            
        elif request.regression_type == "logistic":
            result = analyzer.logistic_regression(
                x_col=request.x_col,
                y_col=request.y_col
            )
            result_dict = result  # Already a dict
            
        else:
            raise ValueError(f"Invalid regression_type: {request.regression_type}. Use 'linear', 'polynomial', or 'logistic'")
        
        return RegressionResponse(
            success=True,
            regression_results=result_dict,
            message=f"{request.regression_type.capitalize()} regression completed successfully",
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in regression: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Regression analysis failed",
                "execution_time": time.time() - start_time
            }
        )


# ============================================================================
# DISTRIBUTION FITTING ENDPOINTS
# ============================================================================

@router.post(
    "/statistical-analysis/distribution-fit",
    response_model=DistributionFitResponse,
    summary="Fit distributions to data",
    description="Fit multiple probability distributions and identify best fit"
)
async def fit_distributions(request: DistributionFitRequest):
    """Fit probability distributions to data."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        results = analyzer.fit_distributions(
            column=request.column,
            distributions=request.distributions
        )
        
        # Convert results to ensure all types are JSON-serializable
        results = convert_results_to_dict(results)
        
        # Find best distribution
        best_dist = None
        for dist_name, result in results.items():
            if result.get('is_best_fit', False):
                best_dist = dist_name
                break
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "fit_results": results,
                "best_distribution": best_dist,
                "message": f"Distribution fitting completed for column '{request.column}'",
                "execution_time": time.time() - start_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error in distribution fitting: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Distribution fitting failed",
                "execution_time": time.time() - start_time
            }
        )


# ============================================================================
# TIME SERIES ANALYSIS ENDPOINTS
# ============================================================================

@router.post(
    "/statistical-analysis/stationarity-test",
    response_model=TimeSeriesAnalysisResponse,
    summary="Test time series stationarity",
    description="Perform Augmented Dickey-Fuller test for stationarity"
)
async def test_stationarity(request: StationarityTestRequest):
    """Test time series stationarity."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        result = analyzer.test_stationarity(
            column=request.column,
            alpha=request.alpha
        )
        
        return TimeSeriesAnalysisResponse(
            success=True,
            analysis_results=result,
            message=f"Stationarity test completed for column '{request.column}'",
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in stationarity test: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Stationarity test failed",
                "execution_time": time.time() - start_time
            }
        )


@router.post(
    "/statistical-analysis/granger-causality",
    response_model=TimeSeriesAnalysisResponse,
    summary="Test Granger causality",
    description="Test if one time series Granger-causes another"
)
async def test_granger_causality(request: GrangerCausalityRequest):
    """Test Granger causality between time series."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        result = analyzer.granger_causality(
            cause_col=request.cause_col,
            effect_col=request.effect_col,
            max_lag=request.max_lag,
            alpha=request.alpha
        )
        
        return TimeSeriesAnalysisResponse(
            success=True,
            analysis_results=result,
            message=f"Granger causality test completed",
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in Granger causality test: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Granger causality test failed",
                "execution_time": time.time() - start_time
            }
        )


# ============================================================================
# SUMMARY STATISTICS ENDPOINTS
# ============================================================================

@router.post(
    "/statistical-analysis/summary-statistics",
    response_model=SummaryStatisticsResponse,
    summary="Get comprehensive summary statistics",
    description="Get detailed statistical summary including mean, std, skewness, kurtosis, etc."
)
async def get_summary_statistics(request: SummaryStatisticsRequest):
    """Get comprehensive summary statistics."""
    start_time = time.time()
    
    try:
        session = get_session_from_request(request.session_id)
        analyzer = AdvancedStatisticalAnalysis(session.dataframe)
        
        stats = analyzer.get_summary_statistics(request.column)
        
        return SummaryStatisticsResponse(
            success=True,
            statistics=stats,
            message=f"Summary statistics computed for column '{request.column}'",
            execution_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in summary statistics: {e}", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "message": "Summary statistics computation failed",
                "execution_time": time.time() - start_time
            }
        )
