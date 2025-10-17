"""
Pydantic schemas for session management API endpoints.
Defines request/response models for data validation.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
import uuid


class SessionStartResponse(BaseModel):
    """Response model for session creation."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="Success message")
    data_preview: Dict[str, Any] = Field(..., description="Dataset preview and metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "3ed5eaa1-4838-4316-9b95-216fd16fd9dd",
                "message": "Session started successfully. Uploaded file 'sales.csv' with 400 rows and 5 columns.",
                "data_preview": {
                    "shape": [400, 5],
                    "columns": ["date", "product", "sales", "region", "profit"],
                    "dtypes": {
                        "date": "object",
                        "product": "object", 
                        "sales": "int64",
                        "region": "object",
                        "profit": "float64"
                    },
                    "sample_data": [
                        {
                            "date": "2024-01-01",
                            "product": "Widget A",
                            "sales": 1200,
                            "region": "North",
                            "profit": 150.50
                        }
                    ],
                    "missing_values": {
                        "date": 0,
                        "product": 0,
                        "sales": 0,
                        "region": 0,
                        "profit": 15
                    }
                }
            }
        }


class QueryRequest(BaseModel):
    """Request model for data analysis queries."""
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., min_length=1, max_length=500, description="Natural language query")
    conversation_context: Optional[str] = Field(None, description="Previous conversation context")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "3ed5eaa1-4838-4316-9b95-216fd16fd9dd",
                "query": "Show me sales by region",
                "conversation_context": "Previous: Analyzed data overview"
            }
        }


class InsightData(BaseModel):
    """Data model for insight responses."""
    response_type: Literal["insight"] = Field("insight", description="Response type discriminator")
    summary: str = Field(..., description="Dataset summary")
    key_insights: List[str] = Field(..., description="Key findings from analysis")
    data_quality: Dict[str, str] = Field(..., description="Data quality assessment")
    suggested_queries: List[str] = Field(..., description="Suggested follow-up queries")


class PlotData(BaseModel):
    """Data model for plot responses."""
    response_type: Literal["plot"] = Field("plot", description="Response type discriminator")
    type: str = Field(..., description="Chart type (plotly)")
    data: Dict[str, Any] = Field(..., description="Plotly figure JSON")
    chart_type: str = Field(..., description="Specific chart type (bar, line, etc.)")
    title: str = Field(..., description="Chart title")


class StatisticsData(BaseModel):
    """Data model for statistics responses."""
    response_type: Literal["statistics"] = Field("statistics", description="Response type discriminator")
    calculation: str = Field(..., description="Type of calculation performed")
    results: Dict[str, Any] = Field(..., description="Statistical results")
    summary_stats: Optional[Dict[str, float]] = Field(None, description="Summary statistics")
    interpretation: str = Field(..., description="Plain English interpretation")


class TextData(BaseModel):
    """Data model for text responses."""
    response_type: Literal["text"] = Field("text", description="Response type discriminator")
    text: str = Field(..., description="Text response")
    message: str = Field(..., description="Response message")


class ErrorData(BaseModel):
    """Data model for error responses."""
    response_type: Literal["error"] = Field("error", description="Response type discriminator")
    error: str = Field(..., description="Error message")
    message: str = Field(..., description="Error description")
    error_type: Optional[str] = Field(None, description="Type of error")


class QueryResponse(BaseModel):
    """Response model for data analysis queries."""
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Original query")
    response_type: str = Field(..., description="Type of response (insight, plot, statistics)")
    data: Union[InsightData, PlotData, StatisticsData, TextData, ErrorData] = Field(..., description="Response data")
    message: str = Field(..., description="Response message")
    execution_time: Optional[float] = Field(None, description="Query execution time in seconds")
    
    @validator('response_type')
    def validate_response_type(cls, v):
        """Validate response type."""
        allowed_types = ['insight', 'plot', 'statistics', 'text', 'error']
        if v not in allowed_types:
            raise ValueError(f'Response type must be one of: {allowed_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "3ed5eaa1-4838-4316-9b95-216fd16fd9dd",
                "query": "Show sales by region",
                "response_type": "plot",
                "data": {
                    "type": "plotly",
                    "data": {"data": [], "layout": {}, "config": {}},
                    "chart_type": "bar",
                    "title": "Sales by Region"
                },
                "message": "Visualization created successfully",
                "execution_time": 1.23
            }
        }


class SessionInfo(BaseModel):
    """Model for session information."""
    session_id: str = Field(..., description="Session identifier")
    filename: str = Field(..., description="Original CSV filename")
    shape: List[int] = Field(..., description="Dataset shape [rows, columns]")
    columns: List[str] = Field(..., description="Column names")
    created_at: str = Field(..., description="Session creation timestamp")
    message_count: int = Field(..., description="Number of messages in conversation")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "3ed5eaa1-4838-4316-9b95-216fd16fd9dd",
                "filename": "sales.csv",
                "shape": [400, 5],
                "columns": ["date", "product", "sales", "region", "profit"],
                "created_at": "2025-10-17T10:30:00Z",
                "message_count": 5
            }
        }


class SessionDeleteResponse(BaseModel):
    """Response model for session deletion."""
    message: str = Field(..., description="Deletion confirmation message")
    session_id: str = Field(..., description="Deleted session identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Session deleted successfully",
                "session_id": "3ed5eaa1-4838-4316-9b95-216fd16fd9dd"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    session_id: Optional[str] = Field(None, description="Session ID if applicable")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid CSV format: No columns to parse from file",
                "status_code": 400,
                "session_id": "3ed5eaa1-4838-4316-9b95-216fd16fd9dd"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    debug: bool = Field(..., description="Debug mode status")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "PyData Assistant Backend",
                "version": "2.0",
                "debug": True,
                "timestamp": "2025-10-17T14:00:00Z"
            }
        }