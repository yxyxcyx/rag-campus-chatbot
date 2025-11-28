# logging_config.py

"""
Structured Logging Configuration

Provides production-ready logging with:
- Structured JSON output for cloud environments (CloudWatch, Datadog, etc.)
- Consistent timestamp and module name formatting
- Configurable log levels
- Request context injection
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

import json


# Context variable for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID in context. Generates one if not provided."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON format for easy parsing by log aggregation services.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add module/function info
        log_entry["module"] = record.module
        log_entry["function"] = record.funcName
        log_entry["line"] = record.lineno
        
        # Add any extra fields passed to the logger
        if hasattr(record, "extra_data") and record.extra_data:
            log_entry["data"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output during development.
    
    Format: TIMESTAMP | LEVEL | MODULE | MESSAGE [DATA]
    """
    
    LEVEL_COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        module = record.module
        message = record.getMessage()
        
        # Add request ID if available
        request_id = get_request_id()
        rid_str = f"[{request_id}] " if request_id else ""
        
        # Add extra data if present
        extra_str = ""
        if hasattr(record, "extra_data") and record.extra_data:
            extra_str = f" | {record.extra_data}"
        
        # Color the level name
        if self.use_colors and level in self.LEVEL_COLORS:
            colored_level = f"{self.LEVEL_COLORS[level]}{level:8}{self.RESET}"
        else:
            colored_level = f"{level:8}"
        
        log_line = f"{timestamp} | {colored_level} | {module:20} | {rid_str}{message}{extra_str}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"
        
        return log_line


class StructuredLogger(logging.Logger):
    """
    Extended logger that supports structured metadata.
    
    Usage:
        logger.info("Query received", query_length=42, user_id="abc123")
    """
    
    def _log_with_extra(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        **kwargs
    ) -> None:
        """Internal log method that handles extra data."""
        # Extract extra data from kwargs
        extra = {"extra_data": kwargs} if kwargs else {}
        
        super()._log(
            level, msg, args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel + 1
        )
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message with optional structured data."""
        self._log_with_extra(logging.DEBUG, msg, args, stacklevel=2, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message with optional structured data."""
        self._log_with_extra(logging.INFO, msg, args, stacklevel=2, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message with optional structured data."""
        self._log_with_extra(logging.WARNING, msg, args, stacklevel=2, **kwargs)
    
    def error(self, msg: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Log error message with optional structured data."""
        self._log_with_extra(
            logging.ERROR, msg, args,
            exc_info=exc_info,
            stacklevel=2,
            **kwargs
        )
    
    def critical(self, msg: str, *args, exc_info: bool = False, **kwargs) -> None:
        """Log critical message with optional structured data."""
        self._log_with_extra(
            logging.CRITICAL, msg, args,
            exc_info=exc_info,
            stacklevel=2,
            **kwargs
        )


# Register our custom logger class
logging.setLoggerClass(StructuredLogger)


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    app_name: str = "rag-chatbot"
) -> StructuredLogger:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON for production. If False, human-readable for dev.
        app_name: Application name for the root logger
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(app_name)
    logger.__class__ = StructuredLogger
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Choose formatter based on environment
    if json_output:
        formatter = StructuredFormatter()
    else:
        formatter = ConsoleFormatter(use_colors=True)
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> StructuredLogger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance for the module
    """
    # Create child logger under the app logger
    logger = logging.getLogger(f"rag-chatbot.{name}")
    logger.__class__ = StructuredLogger
    return logger
