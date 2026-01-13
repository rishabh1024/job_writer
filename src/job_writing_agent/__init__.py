"""
Job Application Writer Package

A modular, well-structured package for creating tailored job applications
using LangChain and LangGraph with LangSmith observability.
"""

__version__ = "0.1.0"

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.addHandler(logging.FileHandler(log_dir / "job_writer.log", mode="a"))
logger.info(
    "Logger initialized. Writing to %s", Path(__file__).parent / "job_writer.log"
)

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"


def _is_interactive():
    """Check if we're running in an interactive environment."""
    return sys.stdin.isatty()


def _set_env(var: str):
    """Set environment variable - only prompt if running interactively."""
    if not os.environ.get(var):
        if _is_interactive():
            from getpass import getpass
            os.environ[var] = getpass(f"{var}: ")
            logger.info(f"{var} set interactively")
        else:
            logger.warning(f"{var} is not set and running non-interactively. Skipping.")


if env_path.exists():
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=True)
else:
    logger.warning(
        ".env file not found at %s. Using system environment variables.", env_path
    )


# List of environment variables to check
environment_key_array = [
    "TAVILY_API_KEY",
    "GEMINI_API_KEY",
    "PINECONE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGSMITH_API_KEY",
    "OPENROUTER_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "CEREBRAS_API_KEY",
]


def load_environment_variables(key_array):
    """Load environment variables, warn if missing."""
    missing_keys = []
    for key in key_array:
        if not os.getenv(key):
            logger.warning(f"{key} environment variable is not set.")
            missing_keys.append(key)
            _set_env(key)
    
    if missing_keys and not _is_interactive():
        logger.warning(f"Missing environment variables (non-interactive mode): {missing_keys}")


# Check for critical environment variables
load_environment_variables(environment_key_array)

# Enable LangSmith tracing if API key is set
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"

__all__ = ["job_app_graph", "workflows/research_workflow"]
