"""
FastAPI router for session management endpoints.
Handles CSV uploads, data analysis queries, and session lifecycle.
"""

import os
import logging
import io
import pandas as pd
from typing import List, Dict, Any, Optional

from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile, 
    File, 
    Depends,
    Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.simple_config import settings
from app.services.session_manager import (
    create_session, 
    get_session, 
    delete_session_data,
    list_sessions,
    Session
)
from app.services.data_analysis import (
    get_data_preview,
    process_query_with_llm,
    convert_numpy_types
)
from app.api.v1.schemas.session import (
    SessionStartResponse, 
    QueryRequest, 
    QueryResponse,
    SessionInfo,
    SessionDeleteResponse,
    ErrorResponse
)


# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


# Dependency to get session
def get_session_from_request(session_id: str) -> Session:
    """Get session from request, raising HTTPException if not found."""
    try:
        return get_session(session_id)
    except ValueError as e:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post(
    "/session/start",
    response_model=SessionStartResponse,
    summary="Start a new analysis session",
    description="Upload a CSV file to begin a new data analysis session.",
    responses={
        200: {"description": "Session created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file format or size"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def start_session(
    file: UploadFile = File(
        ..., 
        description="CSV file to analyze (max 16MB)",
    )
):
    """
    Handles CSV file upload and session initialization.
    """
    # Validate file extension
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_file_type",
                "message": "Only .csv files are allowed.",
                "status_code": 400
            }
        )

    # Read file content
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "file_read_error",
                "message": "Could not read the uploaded file.",
                "status_code": 500
            }
        )
    
    # Create a new session
    try:
        session_id, df = create_session(contents)
        preview = get_data_preview(df, file.filename)
        
        return SessionStartResponse(
            session_id=session_id,
            message=f"Session started successfully. Uploaded file '{file.filename}' with {preview['shape'][0]} rows and {preview['shape'][1]} columns.",
            data_preview=preview
        )
    except ValueError as e:
        logger.error(f"Error creating session: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": "session_creation_failed",
                "message": str(e),
                "status_code": 400
            }
        )
    except Exception as e:
        logger.error(f"Unhandled exception during session start: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred during session creation.",
                "status_code": 500
            }
        )


@router.post(
    "/session/query",
    response_model=QueryResponse,
    summary="Submit a query for analysis",
    description="Send a natural language query for the specified session.",
    responses={
        200: {"description": "Query processed successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def perform_query(
    request: QueryRequest
):
    """
    Processes a natural language query against the session's dataset.
    """
    try:
        # Get the session
        session = get_session_from_request(request.session_id)
        
        response_data, message, exec_time = await process_query_with_llm(
            session=session,
            query=request.query
        )
        
        response = QueryResponse(
            session_id=session.session_id,
            query=request.query,
            response_type=response_data.response_type,
            data=convert_numpy_types(response_data.data),
            message=message,
            execution_time=exec_time
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query for session {session.session_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "query_processing_error",
                "message": str(e),
                "status_code": 500,
                "session_id": session.session_id
            }
        )

@router.get(
    "/sessions",
    response_model=List[SessionInfo],
    summary="List all active sessions",
    description="Get a list of all current analysis sessions.",
    responses={
        200: {"description": "List of active sessions"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_all_sessions():
    """
    Returns information about all active sessions.
    """
    try:
        sessions = list_sessions()
        return [
            SessionInfo(
                session_id=s.session_id,
                filename=os.path.basename(s.metadata.get('filename', 'Unknown')),
                shape=s.metadata.get('shape', [0, 0]),
                columns=s.metadata.get('columns', []),
                created_at=s.metadata.get('created_at', ''),
                message_count=len(s.conversation_history)
            ) for s in sessions
        ]
    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "session_list_error",
                "message": "Failed to retrieve session list.",
                "status_code": 500
            }
        )

@router.get(
    "/session/{session_id}",
    response_model=SessionInfo,
    summary="Get session information",
    description="Retrieve details about a specific session.",
    responses={
        200: {"description": "Session details"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def get_session_info(
    session_id: str
):
    """
    Returns detailed information about a single session.
    """
    session = get_session_from_request(session_id)
    s = session.metadata
    return SessionInfo(
        session_id=session.session_id,
        filename=os.path.basename(s.get('filename', 'Unknown')),
        shape=s.get('shape', [0, 0]),
        columns=s.get('columns', []),
        created_at=s.get('created_at', ''),
        message_count=len(session.conversation_history)
    )


@router.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    summary="Delete a session",
    description="Delete a session and all its associated data.",
    responses={
        200: {"description": "Session deleted"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def delete_session(
    session_id: str
):
    """
    Deletes a session and its cached data.
    """
    try:
        delete_session_data(session_id)
        return SessionDeleteResponse(
            message="Session deleted successfully",
            session_id=session_id
        )
    except ValueError as e:
        logger.warning(f"Attempted to delete non-existent session: {session_id}")
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": str(e),
                "status_code": 404
            }
        )
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An error occurred while deleting the session.",
                "status_code": 500
            }
        )


# New data models for pagination and export
class DataPageRequest(BaseModel):
    session_id: str
    page: int = 1
    page_size: int = 50
    search: Optional[str] = None
    sort_column: Optional[str] = None
    sort_direction: str = 'asc'

class DataPageResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    total_pages: int

class ExportRequest(BaseModel):
    session_id: str
    format: str = 'csv'  # csv, xlsx, json


@router.post(
    "/session/data",
    response_model=DataPageResponse,
    summary="Get paginated dataset data",
    description="Retrieve a specific page of the dataset with optional search and sorting.",
    responses={
        200: {"description": "Data page retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_dataset_page(request: DataPageRequest):
    """
    Get a paginated view of the dataset with optional search and sorting.
    """
    try:
        session = get_session_from_request(request.session_id)
        df = session.dataframe.copy()
        
        # Apply search filter if provided
        if request.search and request.search.strip():
            search_term = request.search.strip().lower()
            mask = df.astype(str).apply(
                lambda x: x.str.lower().str.contains(search_term, na=False)
            ).any(axis=1)
            df = df[mask]
        
        # Apply sorting if provided
        if request.sort_column and request.sort_column in df.columns:
            ascending = request.sort_direction.lower() == 'asc'
            df = df.sort_values(by=request.sort_column, ascending=ascending)
        
        # Calculate pagination
        total_rows = len(df)
        total_pages = max(1, (total_rows + request.page_size - 1) // request.page_size)
        start_idx = (request.page - 1) * request.page_size
        end_idx = min(start_idx + request.page_size, total_rows)
        
        # Get the page data
        page_df = df.iloc[start_idx:end_idx]
        
        # Convert to records format
        rows = page_df.fillna('').to_dict('records')
        
        # Convert numpy types to JSON serializable
        rows = convert_numpy_types(rows)
        
        return DataPageResponse(
            columns=list(df.columns),
            rows=rows,
            total_rows=total_rows,
            page=request.page,
            page_size=request.page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error getting dataset page for session {request.session_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "data_page_error",
                "message": str(e),
                "status_code": 500
            }
        )


@router.post(
    "/session/export",
    summary="Export dataset",
    description="Export the dataset in various formats (CSV, Excel, JSON).",
    responses={
        200: {"description": "File exported successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        400: {"model": ErrorResponse, "description": "Invalid export format"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def export_dataset(request: ExportRequest):
    """
    Export the dataset in the specified format.
    """
    try:
        session = get_session_from_request(request.session_id)
        df = session.dataframe.copy()
        
        # Create filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if request.format.lower() == 'csv':
            # Export as CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            filename = f"dataset_{timestamp}.csv"
            media_type = "text/csv"
            content = output.getvalue()
            
            return StreamingResponse(
                io.StringIO(content),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
        elif request.format.lower() in ['xlsx', 'excel']:
            # Export as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
            output.seek(0)
            
            filename = f"dataset_{timestamp}.xlsx"
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            return StreamingResponse(
                io.BytesIO(output.getvalue()),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
            
        elif request.format.lower() == 'json':
            # Export as JSON
            json_data = df.to_json(orient='records', indent=2)
            
            filename = f"dataset_{timestamp}.json"
            media_type = "application/json"
            
            return StreamingResponse(
                io.StringIO(json_data),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_format",
                    "message": f"Unsupported export format: {request.format}. Supported formats: csv, xlsx, json",
                    "status_code": 400
                }
            )
            
    except Exception as e:
        logger.error(f"Error exporting dataset for session {request.session_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "export_error",
                "message": str(e),
                "status_code": 500
            }
        )
