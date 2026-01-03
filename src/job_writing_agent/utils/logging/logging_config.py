"""
Logging configuration for the application

This module provides a centralized logging manager that configures
logging once at application startup, ensuring consistent log format
and behavior across all modules.
"""

import logging
import sys
from pathlib import Path
from typing_extensions import Optional


class LoggingManager:
    """
    Centralized logging configuration manager.

    Uses Singleton pattern to ensure logging is configured only once.

    Example:
        >>> manager = LoggingManager()
        >>> manager.configure_logging(log_level=logging.INFO)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("This will be logged consistently")
    """

    _instance: Optional["LoggingManager"] = None
    _configured: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._configured = False
        return cls._instance

    def configure_logging(
        self,
        log_level: int = logging.INFO,
        log_file: Optional[Path] = None,
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
    ) -> None:
        """
        Configure logging for the entire application.

        This should be called once at application startup (e.g., in main()).
        Subsequent calls are ignored if already configured.

        Args:
            log_level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
            log_file: Optional path to log file. If None, logs only to console.
            log_format: Optional custom format string. Default includes timestamp, level, module, message.
            date_format: Optional date format string. Default: "%Y-%m-%d %H:%M:%S"

        Example:
            >>> manager = LoggingManager()
            >>> manager.configure_logging(
            ...     log_level=logging.INFO,
            ...     log_file=Path("logs/app.log")
            ... )
        """
        if self._configured:
            # Already configured - don't reconfigure
            return

        # Default format: [2025-01-15 10:30:45] INFO    module_name: message
        if log_format is None:
            log_format = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Console handler (always add)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (if log_file specified)
        if log_file:
            # Create log directory if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        self._configured = True

        # Log that logging is configured
        logger = logging.getLogger(__name__)
        logger.info(
            f"Logging configured: level={logging.getLevelName(log_level)}, "
            f"file={'enabled' if log_file else 'disabled'}"
        )

    def is_configured(self) -> bool:
        """Check if logging has been configured."""
        return self._configured


# Convenience function for easy access
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    This is a convenience function that ensures consistent logger creation.
    Use this instead of logging.getLogger(__name__) for consistency.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)
