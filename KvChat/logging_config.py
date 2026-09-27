"""
Centralized logging configuration for KvChat.

This module provides a unified logging setup with support for:
- Colored console output (when ANSI is supported)
- Rotating file handlers (automatic log rotation)
- Configurable log levels and formats
- Per-module logger access
"""

import logging
import logging.handlers
import sys
from typing import Optional


# ANSI color codes for terminal output
class _ColorCodes:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta


class _ColorFormatter(logging.Formatter):
    """Custom formatter with optional colored output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: _ColorCodes.DEBUG,
        logging.INFO: _ColorCodes.INFO,
        logging.WARNING: _ColorCodes.WARNING,
        logging.ERROR: _ColorCodes.ERROR,
        logging.CRITICAL: _ColorCodes.CRITICAL,
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_color: bool = True):
        """
        Initialize color formatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            use_color: Whether to use ANSI colors (auto-detect if True)
        """
        super().__init__(fmt, datefmt)
        self.use_color = use_color and self._supports_ansi()
    
    @staticmethod
    def _supports_ansi() -> bool:
        """Check if terminal supports ANSI color codes."""
        # Check for common ANSI support indicators
        if not hasattr(sys.stdout, 'isatty'):
            return False
        return sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        if self.use_color:
            levelname = record.levelname
            color = self.LEVEL_COLORS.get(record.levelno, _ColorCodes.RESET)
            record.levelname = f"{color}{levelname:8s}{_ColorCodes.RESET}"
        
        result = super().format(record)
        return result


def setup_logging(
    level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    log_file: Optional[str] = None,
    use_color: bool = True,
) -> logging.Logger:
    """
    Set up centralized logging configuration for the application.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Log message format string
        date_format: Date format string for timestamps
        log_file: Optional log file path (enables file logging)
        use_color: Whether to use colors in console output
    
    Returns:
        Root logger configured with the specified settings
    
    Example:
        >>> logger = setup_logging(level=logging.INFO, log_file='app.log')
        >>> logger.info("Application started")
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = _ColorFormatter(log_format, date_format, use_color)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=10
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance for the module
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)
