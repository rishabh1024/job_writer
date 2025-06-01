"""
Job Application Writer Package

A modular, well-structured package for creating tailored job applications
using LangChain and LangGraph with LangSmith observability.
"""

__version__ = "0.1.0"

import os, getpass
import logging
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
logger.addHandler(logging.FileHandler(log_dir / 'job_writer.log', mode='a'))
logger.info("Logger initialized. Writing to %s", Path(__file__).parent / 'job_writer.log')

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        logger.info(f"{var} set to {os.environ[var]}")

if env_path.exists():
    logger.info("Loading environment variables from %s",  env_path)
    load_dotenv(dotenv_path=env_path, override=True)
else:
    logger.warning(".env file not found at %s. Using system environment variables.", env_path)

# Check for critical environment variables
if not os.getenv("TAVILY_API_KEY"):
    logger.warning("TAVILY_API_KEY environment variable is not set." \
                    " Failed to get TAVILY_API_KEY at Path %s", env_path)
    _set_env("TAVILY_API_KEY")


if not os.getenv("GEMINI_API_KEY"):
    logger.warning("GEMINI_API_KEY environment variable is not set. " \
                    "Failed to get GEMINI_API_KEY at Path %s", env_path)
    _set_env("GEMINI_API_KEY")


if not os.getenv("PINECONE_API_KEY"):
    logger.warning("PINECONE_API_KEY environment variable is not set." \
                " Failed to get PINECONE_API_KEY at Path %s", env_path)
    _set_env("PINECONE_API_KEY")

if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    logger.warning("LANGFUSE_PUBLIC_KEY environment variable is not set." \
                " Failed to get LANGFUSE_PUBLIC_KEY at Path %s", env_path)
    _set_env("LANGFUSE_PUBLIC_KEY")

if not os.getenv("LANGFUSE_SECRET_KEY"):
    logger.warning("LANGFUSE_SECRET_KEY environment variable is not set." \
                " Failed to get LANGFUSE_SECRET_KEY at Path %s", env_path)
    _set_env("LANGFUSE_SECRET_KEY")


__all__: list[str] = ["job_app_graph", "workflows/research_workflow"]