"""
FastAPI router for session management endpoints.
Handles CSV uploads, data analysis queries, and session lifecycle.
"""

import os
import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile, 
    File, 
    Depends,
    Request
)
from fastapi.responses import JSONResponse

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


@router.get(
    "/session/{session_id}/data",
    summary="Get full dataset with pagination",
    description="Retrieve the complete dataset with pagination support",
    responses={
        200: {"description": "Dataset retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def get_full_data(
    session_id: str,
    page: int = 1,
    page_size: int = 100
):
    """
    Returns paginated full dataset from the session.
    """
    try:
        session = get_session_from_request(session_id)
        df = session.dataframe
        
        # Calculate pagination
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        # Get paginated data and replace NaN/Inf with None for JSON compatibility
        page_df = df.iloc[start_idx:end_idx].copy()
        page_df = page_df.replace([np.inf, -np.inf], np.nan)  # Convert Inf to NaN
        page_df = page_df.where(pd.notna(page_df), None)  # Convert NaN to None
        page_data = page_df.to_dict('records')
        
        return {
            "session_id": session_id,
            "data": convert_numpy_types(page_data),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "total_pages": total_pages,
                "start_row": start_idx + 1,
                "end_row": end_idx
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving data for session {session_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "data_retrieval_error",
                "message": str(e),
                "status_code": 500
            }
        )

