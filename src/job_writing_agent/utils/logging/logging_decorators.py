"""
Simple decorators for logging.

These decorators add logging behavior without cluttering your function code.
Keep it simple - just the essentials.
"""

import functools
import logging
import time
from typing import Callable, TypeVar

# Type variable for function signatures
F = TypeVar("F", bound=Callable)

logger = logging.getLogger(__name__)


def log_execution(func: F) -> F:
    """
    Simple decorator to log when a function starts and finishes.

    Logs entry, exit, and how long it took.

    Example:
        >>> @log_execution
        >>> def process_data(data: str) -> str:
        ...     return data.upper()
        >>> process_data("hello")
        # Logs: "Entering process_data" ... "Exiting process_data (took 0.001s)"
    """

    @functools.wraps(func)
    def log_execution_wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Entering {func_name}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Exiting {func_name} (took {elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func_name} failed after {elapsed:.3f}s: {e}", exc_info=True)
            raise

    return log_execution_wrapper


def log_async(func: F) -> F:
    """
    Simple decorator for async functions - logs entry, exit, and timing.

    Example:
        >>> @log_async
        >>> async def fetch_data(url: str) -> dict:
        ...     return await http.get(url)
    """

    @functools.wraps(func)
    async def log_async_wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Entering async {func_name}")

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Exiting async {func_name} (took {elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func_name} failed after {elapsed:.3f}s: {e}", exc_info=True)
            raise

    return log_async_wrapper


def log_errors(func: F) -> F:
    """
    Simple decorator to catch and log exceptions.

    Logs the error, then re-raises it so your code still fails normally.

    Example:
        >>> @log_errors
        >>> def risky_operation():
        ...     raise ValueError("Something went wrong")
        >>> risky_operation()
        # Logs the error, then raises it
    """

    @functools.wraps(func)
    def log_errors_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise

    return log_errors_wrapper
