import argparse
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_MODEL = "allenai/olmo-3.1-32b-think:free"
DEFAULT_CONTENT_TYPE = "cover_letter"


def readable_file(path: str) -> str:
    """
    Validate that the file exists and has a supported extension.

    Args:
        path: File path to validate

    Returns:
        Original path string if valid

    Raises:
        ArgumentTypeError: If file doesn't exist or has unsupported extension
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not path.lower().endswith((".pdf", ".md", ".json", ".txt")):
        raise argparse.ArgumentTypeError(
            "Only text files (.txt, .md, .pdf, .json) are supported."
        )
    return path


def valid_temp(temp: str) -> float:
    """
    Ensure temperature is within a reasonable range.

    Args:
        temp: Temperature value as string

    Returns:
        Temperature as float

    Raises:
        ArgumentTypeError: If temperature is outside valid range [0, 2]
    """
    value = float(temp)
    if not (0 <= value <= 2):
        raise argparse.ArgumentTypeError("Temperature must be between 0 and 2.")
    return value


def is_valid_url(
    job_posting: str, allowed_statuses: Iterable[int] | None = None
) -> str:
    """
    Validate that a URL is reachable and returns an acceptable HTTP status.

    Defaults to any 2xx or 3xx response (common successful codes).

    Args:
        job_posting: The URL for the job posting
        allowed_statuses: Specific status codes that are considered valid.
            If None (default), any 200-399 status is accepted.

    Returns:
        URL of the job posting if successful, error message if failed
    """
    if allowed_statuses is None:
        # All 2xx and 3xx responses are considered “valid”
        allowed_statuses = range(200, 400)

    try:
        response = requests.get(
            job_posting, timeout=30, allow_redirects=True, stream=True
        )
        response.raise_for_status()
        return job_posting
    except requests.exceptions.RequestException as e:
        return f"Error: {e.response.text if e.response else 'Unknown error'}"


def handle_cli() -> argparse.Namespace:
    """
    Parse and validate CLI arguments for job application generator.

    Returns:
        Parsed command-line arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="""Assist the candidate in writing content for
        job application such as answering to question in application
        process, cover letters and more."""
    )
    parser.add_argument(
        "-r",
        "--resume",
        required=True,
        metavar="resume",
        type=readable_file,
        help="Relative/Absolute path to resume file in pdf, text, markdown format.",
    )
    parser.add_argument(
        "-j",
        "--job_posting",
        required=True,
        metavar="job_posting",
        type=is_valid_url,
        help="URL to job posting or paste raw text of job description text.",
    )
    parser.add_argument(
        "-t",
        "--content_type",
        default=DEFAULT_CONTENT_TYPE,
        choices=["cover_letter", "bullets", "linkedin_note"],
        help="Type of application material to generate (default: cover_letter).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help="Model to use (default: qwen/qwen3-4b:free).",
    )
    parser.add_argument(
        "--temp",
        type=valid_temp,
        default=0.2,
        metavar="FLOAT",
        help="Temperature for generation, 0-2 (default: 0.7).",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    return parser.parse_args()
