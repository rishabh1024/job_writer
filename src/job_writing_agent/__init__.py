"""
Job Application Writer Package

A modular, well-structured package for creating tailored job applications
using LangChain and LangGraph with LangSmith observability.
"""

__version__ = "0.1.0"

import os
from getpass import getpass
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


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"{var}: ")
        logger.info(f"{var} set to {os.environ[var]}")


if env_path.exists():
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=True)
else:
    logger.warning(
        ".env file not found at %s. Using system environment variables.", env_path
    )

# Check for critical environment variables
if not os.getenv("TAVILY_API_KEY"):
    logger.warning(
        "TAVILY_API_KEY environment variable is not set."
        " Failed to get TAVILY_API_KEY at Path %s",
        env_path,
    )
    _set_env("TAVILY_API_KEY")


if not os.getenv("GEMINI_API_KEY"):
    logger.warning(
        "GEMINI_API_KEY environment variable is not set. "
        "Failed to get GEMINI_API_KEY at Path %s",
        env_path,
    )
    _set_env("GEMINI_API_KEY")


if not os.getenv("PINECONE_API_KEY"):
    logger.warning(
        "PINECONE_API_KEY environment variable is not set."
        " Failed to get PINECONE_API_KEY at Path %s",
        env_path,
    )
    _set_env("PINECONE_API_KEY")

if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    logger.warning(
        "LANGFUSE_PUBLIC_KEY environment variable is not set."
        " Failed to get LANGFUSE_PUBLIC_KEY at Path %s",
        env_path,
    )
    _set_env("LANGFUSE_PUBLIC_KEY")

if not os.getenv("LANGFUSE_SECRET_KEY"):
    logger.warning(
        "LANGFUSE_SECRET_KEY environment variable is not set."
        " Failed to get LANGFUSE_SECRET_KEY at Path %s",
        env_path,
    )
    _set_env("LANGFUSE_SECRET_KEY")

if not os.getenv("LANGSMITH_API_KEY"):
    logger.warning(
        "LANGSMITH_API_KEY environment variable is not set."
        " Failed to get LANGSMITH_API_KEY at Path %s",
        env_path,
    )
    _set_env("LANGSMITH_API_KEY")

if not os.getenv("OPENROUTER_API_KEY"):
    logger.warning(
        "OPENROUTER_API_KEY environment variable is not set."
        " Failed to get OPENROUTER_API_KEY at Path %s",
        env_path,
    )
    _set_env("OPENROUTER_API_KEY")

if not os.getenv("LANGSMITH_PROJECT"):
    logger.warning(
        "LANGSMITH_PROJECT environment variable is not set."
        " Failed to get LANGSMITH_PROJECT at Path %s",
        env_path,
    )
    _set_env("LANGSMITH_PROJECT")

if not os.getenv("LANGSMITH_ENDPOINT"):
    logger.warning(
        "LANGSMITH_ENDPOINT environment variable is not set."
        " Failed to get LANGSMITH_ENDPOINT at Path %s",
        env_path,
    )
    _set_env("LANGSMITH_ENDPOINT")

if not os.getenv("CEREBRAS_API_KEY"):
    logger.warning(
        "CEREBRAS_API_KEY environment variable is not set."
        " Failed to get CEREBRAS_API_KEY at Path %s",
        env_path,
    )
    _set_env("CEREBRAS_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"

__all__: list[str] = ["job_app_graph", "workflows/research_workflow"]


"""
Job Application Writer Package

A modular, well-structured package for creating tailored job applications
using LangChain and LangGraph with LangSmith observability.
"""

__version__ = "0.1.0"

import os
import getpass
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.addHandler(logging.FileHandler(log_dir / "job_writer.log", mode="a"))
logger.info(
    "Logger initialized. Writing to %s", Path(__file__).parent / "job_writer.log"
)

env_path = Path(__file__).parent / ".env"


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
        logger.info(f"{var} set to {os.environ[var]}")


def load_environment_variables(key_array):
    for key in key_array:
        if not os.getenv(key):
            logger.warning(
                f"{key} environment variable is not set. Failed to get {key} at Path {env_path}"
            )
            _set_env(key)


if env_path.exists():
    logger.info("Loading environment variables from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=True)
else:
    logger.warning(
        ".env file not found at %s. Using system environment variables.", env_path
    )


environment_key_array = [
    "TAVILY_API_KEY",
    "GEMINI_API_KEY",
    "PINECONE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]
# Check for critical environment variables
load_environment_variables(environment_key_array)

__all__ = ["job_app_graph", "workflows/research_workflow"]
