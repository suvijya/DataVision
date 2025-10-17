"""
Session management service for handling data analysis sessions.
Manages session lifecycle, data storage, and conversation history.
"""

import os
import json
import pickle
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
from io import StringIO

from app.core.simple_config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in the conversation history."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    query_type: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class Session:
    """Represents an analysis session with data and conversation history."""
    session_id: str
    dataframe: pd.DataFrame
    metadata: Dict[str, Any]
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def add_message(self, role: str, content: str, query_type: Optional[str] = None, execution_time: Optional[float] = None):
        """Add a message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query_type=query_type,
            execution_time=execution_time
        )
        self.conversation_history.append(message)
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        
        # Keep conversation history within limits
        if len(self.conversation_history) > settings.MAX_CONVERSATION_MESSAGES:
            # Remove oldest messages, keeping some context
            self.conversation_history = self.conversation_history[-settings.MAX_CONVERSATION_MESSAGES:]
    
    def get_conversation_context(self, max_tokens: Optional[int] = None) -> str:
        """Get conversation context for LLM with token limit."""
        context_parts = []
        total_tokens = 0
        max_tokens = max_tokens or settings.CONVERSATION_MEMORY_LIMIT
        
        # Start from recent messages and work backwards
        for message in reversed(self.conversation_history):
            message_text = f"{message.role}: {message.content}"
            # Rough token estimation (1 token ≈ 4 characters)
            message_tokens = len(message_text) // 4
            
            if total_tokens + message_tokens > max_tokens:
                break
                
            context_parts.insert(0, message_text)
            total_tokens += message_tokens
        
        return "\n\n".join(context_parts)


# In-memory session storage
_active_sessions: Dict[str, Session] = {}


def create_session(csv_content: bytes) -> Tuple[str, pd.DataFrame]:
    """
    Create a new analysis session from CSV content.
    
    Args:
        csv_content: Raw CSV file bytes
        
    Returns:
        Tuple of (session_id, dataframe)
        
    Raises:
        ValueError: If CSV is invalid or empty
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Convert bytes to string and read CSV
        csv_string = csv_content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))
        
        # Validate dataframe
        if df.empty:
            raise ValueError("CSV file is empty")
        
        if len(df.columns) == 0:
            raise ValueError("CSV file has no columns")
            
        # Clean column names (remove leading/trailing whitespace)
        df.columns = df.columns.str.strip()
        
        # Basic data type inference
        # Try to convert numeric columns
        for column in df.columns:
            if df[column].dtype == 'object':
                # Try to convert to numeric (will fail silently if not possible)
                converted = pd.to_numeric(df[column], errors='ignore')
                if converted.dtype != 'object':
                    df[column] = converted
                else:
                    # Try to convert to datetime
                    try:
                        df[column] = pd.to_datetime(df[column], errors='ignore', infer_datetime_format=True)
                    except:
                        pass  # Keep as string if conversion fails
        
        # Create session metadata
        metadata = {
            'filename': 'uploaded_file.csv',
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage': df.memory_usage(deep=True).sum(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Create session object
        session = Session(
            session_id=session_id,
            dataframe=df,
            metadata=metadata
        )
        
        # Store session
        _active_sessions[session_id] = session
        
        # Save session to cache directory
        _save_session_to_cache(session)
        
        logger.info(f"Created session {session_id} with {df.shape[0]} rows, {df.shape[1]} columns")
        
        return session_id, df
        
    except UnicodeDecodeError:
        raise ValueError("Invalid CSV file encoding. Please ensure file is UTF-8 encoded.")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or contains no data")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating session: {e}", exc_info=True)
        raise ValueError(f"Failed to process CSV file: {str(e)}")


def get_session(session_id: str) -> Session:
    """
    Get an existing session by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session object
        
    Raises:
        ValueError: If session not found
    """
    if session_id in _active_sessions:
        session = _active_sessions[session_id]
        session.last_accessed = datetime.now(timezone.utc).isoformat()
        return session
    
    # Try to load from cache
    session = _load_session_from_cache(session_id)
    if session:
        _active_sessions[session_id] = session
        session.last_accessed = datetime.now(timezone.utc).isoformat()
        return session
    
    raise ValueError(f"Session {session_id} not found")


def list_sessions() -> List[Session]:
    """
    Get all active sessions.
    
    Returns:
        List of all session objects
    """
    # Load any cached sessions not in memory
    _load_all_cached_sessions()
    
    return list(_active_sessions.values())


def delete_session_data(session_id: str):
    """
    Delete a session and its cached data.
    
    Args:
        session_id: Session identifier
        
    Raises:
        ValueError: If session not found
    """
    if session_id not in _active_sessions:
        # Try loading from cache first
        try:
            get_session(session_id)
        except ValueError:
            raise ValueError(f"Session {session_id} not found")
    
    # Remove from memory
    del _active_sessions[session_id]
    
    # Remove from cache
    cache_file = os.path.join(settings.CACHE_DIR, f"{session_id}.pkl")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.info(f"Deleted session {session_id} and cache file")
    else:
        logger.info(f"Deleted session {session_id} (no cache file found)")


def cleanup_expired_sessions():
    """
    Clean up sessions that have exceeded the timeout.
    This should be called periodically by a background task.
    """
    current_time = datetime.now(timezone.utc)
    expired_sessions = []
    
    for session_id, session in _active_sessions.items():
        last_accessed = datetime.fromisoformat(session.last_accessed.replace('Z', '+00:00'))
        age_seconds = (current_time - last_accessed).total_seconds()
        
        if age_seconds > settings.SESSION_TIMEOUT:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        try:
            delete_session_data(session_id)
            logger.info(f"Cleaned up expired session: {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")


def _save_session_to_cache(session: Session):
    """Save session to cache directory."""
    try:
        os.makedirs(settings.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(settings.CACHE_DIR, f"{session.session_id}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(session, f)
            
        logger.debug(f"Saved session {session.session_id} to cache")
        
    except Exception as e:
        logger.error(f"Failed to save session to cache: {e}")


def _load_session_from_cache(session_id: str) -> Optional[Session]:
    """Load session from cache directory."""
    try:
        cache_file = os.path.join(settings.CACHE_DIR, f"{session_id}.pkl")
        
        if not os.path.exists(cache_file):
            return None
            
        with open(cache_file, 'rb') as f:
            session = pickle.load(f)
            
        logger.debug(f"Loaded session {session_id} from cache")
        return session
        
    except Exception as e:
        logger.error(f"Failed to load session from cache: {e}")
        return None


def _load_all_cached_sessions():
    """Load all cached sessions that aren't in memory."""
    try:
        if not os.path.exists(settings.CACHE_DIR):
            return
            
        for filename in os.listdir(settings.CACHE_DIR):
            if filename.endswith('.pkl'):
                session_id = filename[:-4]  # Remove .pkl extension
                
                if session_id not in _active_sessions:
                    session = _load_session_from_cache(session_id)
                    if session:
                        _active_sessions[session_id] = session
                        
    except Exception as e:
        logger.error(f"Error loading cached sessions: {e}")