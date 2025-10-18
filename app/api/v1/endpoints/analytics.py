"""
FastAPI router for advanced analytics endpoints.
Handles ML models, statistical tests, clustering, and dimensionality reduction.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.session_manager import get_session
from app.services.analytics_service import analytics_service
from app.services.data_analysis import convert_numpy_types


logger = logging.getLogger(__name__)
router = APIRouter()


# ==================== REQUEST MODELS ====================

class RegressionRequest(BaseModel):
    """Request model for regression analysis."""
    session_id: str
    target_column: str
    feature_columns: List[str]
    model_type: str = Field(default="linear", description="'linear' or 'logistic'")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class ClusteringRequest(BaseModel):
    """Request model for clustering analysis."""
    session_id: str
    feature_columns: List[str]
    algorithm: str = Field(default="kmeans", description="'kmeans' or 'dbscan'")
    n_clusters: Optional[int] = Field(default=3, ge=2, le=20)
    eps: Optional[float] = Field(default=0.5, description="DBSCAN epsilon")
    min_samples: Optional[int] = Field(default=5, description="DBSCAN min_samples")


class DimensionalityReductionRequest(BaseModel):
    """Request model for dimensionality reduction."""
    session_id: str
    feature_columns: List[str]
    algorithm: str = Field(default="pca", description="'pca' or 'tsne'")
    n_components: int = Field(default=2, ge=2, le=3)
    perplexity: Optional[int] = Field(default=30, description="t-SNE perplexity")


class OutlierDetectionRequest(BaseModel):
    """Request model for outlier detection."""
    session_id: str
    feature_columns: List[str]
    contamination: float = Field(default=0.1, ge=0.01, le=0.5)


class StatisticalTestRequest(BaseModel):
    """Request model for statistical tests."""
    session_id: str
    test_type: str = Field(description="'ttest', 'anova', 'chi_square', or 'correlation'")
    group_column: Optional[str] = None
    value_column: Optional[str] = None
    column1: Optional[str] = None
    column2: Optional[str] = None
    method: Optional[str] = Field(default="pearson", description="Correlation method")


# ==================== ENDPOINTS ====================

@router.post("/analytics/regression")
async def perform_regression(request: RegressionRequest):
    """
    Perform linear or logistic regression analysis.
    
    - **Linear Regression**: Predict continuous values
    - **Logistic Regression**: Binary/multiclass classification
    """
    try:
        # Get session
        session = get_session(request.session_id)
        if not session or session.df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        df = session.df
        
        # Validate columns
        all_cols = [request.target_column] + request.feature_columns
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found: {', '.join(missing_cols)}"
            )
        
        # Perform analysis
        if request.model_type == "linear":
            result = analytics_service.linear_regression(
                df,
                request.target_column,
                request.feature_columns,
                request.test_size
            )
        elif request.model_type == "logistic":
            result = analytics_service.logistic_regression(
                df,
                request.target_column,
                request.feature_columns,
                request.test_size
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {request.model_type}. Use 'linear' or 'logistic'"
            )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.message)
        
        # Convert numpy types for JSON serialization
        result.data = convert_numpy_types(result.data)
        
        return {
            "success": True,
            "analysis_type": result.analysis_type,
            "data": result.data,
            "visualization": result.visualization,
            "message": result.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regression analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/clustering")
async def perform_clustering(request: ClusteringRequest):
    """
    Perform clustering analysis.
    
    - **K-Means**: Partition data into k clusters
    - **DBSCAN**: Density-based clustering (finds arbitrary shapes)
    """
    try:
        # Get session
        session = get_session(request.session_id)
        if not session or session.df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        df = session.df
        
        # Validate columns
        missing_cols = [col for col in request.feature_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found: {', '.join(missing_cols)}"
            )
        
        # Perform analysis
        if request.algorithm == "kmeans":
            result = analytics_service.kmeans_clustering(
                df,
                request.feature_columns,
                request.n_clusters or 3
            )
        elif request.algorithm == "dbscan":
            result = analytics_service.dbscan_clustering(
                df,
                request.feature_columns,
                request.eps or 0.5,
                request.min_samples or 5
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm: {request.algorithm}. Use 'kmeans' or 'dbscan'"
            )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.message)
        
        result.data = convert_numpy_types(result.data)
        
        return {
            "success": True,
            "analysis_type": result.analysis_type,
            "data": result.data,
            "visualization": result.visualization,
            "message": result.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clustering analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/dimensionality-reduction")
async def perform_dimensionality_reduction(request: DimensionalityReductionRequest):
    """
    Perform dimensionality reduction.
    
    - **PCA**: Principal Component Analysis (linear)
    - **t-SNE**: t-Distributed Stochastic Neighbor Embedding (non-linear)
    """
    try:
        # Get session
        session = get_session(request.session_id)
        if not session or session.df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        df = session.df
        
        # Validate columns
        missing_cols = [col for col in request.feature_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found: {', '.join(missing_cols)}"
            )
        
        # Perform analysis
        if request.algorithm == "pca":
            result = analytics_service.pca_analysis(
                df,
                request.feature_columns,
                request.n_components
            )
        elif request.algorithm == "tsne":
            result = analytics_service.tsne_analysis(
                df,
                request.feature_columns,
                request.n_components,
                request.perplexity or 30
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm: {request.algorithm}. Use 'pca' or 'tsne'"
            )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.message)
        
        result.data = convert_numpy_types(result.data)
        
        return {
            "success": True,
            "analysis_type": result.analysis_type,
            "data": result.data,
            "visualization": result.visualization,
            "message": result.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dimensionality reduction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/outliers")
async def detect_outliers(request: OutlierDetectionRequest):
    """
    Detect outliers using Isolation Forest algorithm.
    
    Returns indices and samples of detected outliers.
    """
    try:
        # Get session
        session = get_session(request.session_id)
        if not session or session.df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        df = session.df
        
        # Validate columns
        missing_cols = [col for col in request.feature_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found: {', '.join(missing_cols)}"
            )
        
        # Perform analysis
        result = analytics_service.isolation_forest(
            df,
            request.feature_columns,
            request.contamination
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.message)
        
        result.data = convert_numpy_types(result.data)
        
        return {
            "success": True,
            "analysis_type": result.analysis_type,
            "data": result.data,
            "visualization": result.visualization,
            "message": result.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Outlier detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/statistical-test")
async def perform_statistical_test(request: StatisticalTestRequest):
    """
    Perform statistical tests.
    
    - **T-Test**: Compare means of two groups
    - **ANOVA**: Compare means of 3+ groups
    - **Chi-Square**: Test independence of categorical variables
    - **Correlation**: Analyze relationships between numeric variables
    """
    try:
        # Get session
        session = get_session(request.session_id)
        if not session or session.df is None:
            raise HTTPException(status_code=404, detail="Session not found or no data loaded")
        
        df = session.df
        
        # Perform test based on type
        if request.test_type == "ttest":
            if not request.group_column or not request.value_column:
                raise HTTPException(
                    status_code=400,
                    detail="T-test requires 'group_column' and 'value_column'"
                )
            
            result = analytics_service.ttest(
                df,
                request.group_column,
                request.value_column
            )
            
        elif request.test_type == "anova":
            if not request.group_column or not request.value_column:
                raise HTTPException(
                    status_code=400,
                    detail="ANOVA requires 'group_column' and 'value_column'"
                )
            
            result = analytics_service.anova_test(
                df,
                request.group_column,
                request.value_column
            )
            
        elif request.test_type == "chi_square":
            if not request.column1 or not request.column2:
                raise HTTPException(
                    status_code=400,
                    detail="Chi-square test requires 'column1' and 'column2'"
                )
            
            result = analytics_service.chi_square_test(
                df,
                request.column1,
                request.column2
            )
            
        elif request.test_type == "correlation":
            result = analytics_service.correlation_analysis(
                df,
                request.method or "pearson"
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid test_type: {request.test_type}. "
                       f"Use 'ttest', 'anova', 'chi_square', or 'correlation'"
            )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.message)
        
        result.data = convert_numpy_types(result.data)
        
        return {
            "success": True,
            "analysis_type": result.analysis_type,
            "data": result.data,
            "visualization": result.visualization,
            "message": result.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistical test error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/info")
async def get_analytics_info():
    """Get information about available analytics methods."""
    return {
        "regression": {
            "linear": "Predict continuous values using linear relationships",
            "logistic": "Binary or multiclass classification"
        },
        "clustering": {
            "kmeans": "Partition data into k clusters",
            "dbscan": "Density-based clustering, finds arbitrary shapes"
        },
        "dimensionality_reduction": {
            "pca": "Principal Component Analysis - linear method",
            "tsne": "t-SNE - non-linear method for visualization"
        },
        "outlier_detection": {
            "isolation_forest": "Detect outliers using isolation forest algorithm"
        },
        "statistical_tests": {
            "ttest": "Compare means of two groups",
            "anova": "Compare means of 3+ groups",
            "chi_square": "Test independence of categorical variables",
            "correlation": "Analyze relationships between numeric variables"
        }
    }
