import argparse
import os
from typing import Optional, Any, Iterable

import requests
from requests.exceptions import RequestException


DEFAULT_MODEL = "qwen/qwen3-4b:free"
DEFAULT_CONTENT_TYPE = "cover_letter"


def readable_file(path: str) -> str:
    """Validate and return contents of a readable file."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not path.lower().endswith((".pdf", ".md", ".json", ".txt")):
        raise argparse.ArgumentTypeError(
            "Only text files (.txt, .md, .pdf, .json) are supported."
        )
    return path


def valid_temp(temp: str) -> float:
    """Ensure temperature is within a reasonable range."""
    value = float(temp)
    if not (0 <= value <= 2):
        raise argparse.ArgumentTypeError("Temperature must be between 0 and 2.")
    return value


def is_valid_url(
    job_posting: str, allowed_statuses: Iterable[int] | None = None
) -> bool:
    """
    Returns ``True`` if *url* is reachable and its HTTP status code is in
    `allowed_statuses`.  Defaults to any 2xx or 3xx response (common
    successful codes).

    Parameters
    ----------
    job_posting : str
        The URL for the job posting.
    timeout : float, optional
        Timeout for the request (seconds). Defaults to 10.
    allowed_statuses : Iterable[int] | None, optional
        Specific status codes that are considered “valid”.
        If ``None`` (default) any 200‑399 status is accepted.

    Returns
    -------
    bool
        ``True`` if the URL succeeded, ``False`` otherwise.
    """
    if allowed_statuses is None:
        # All 2xx and 3xx responses are considered “valid”
        allowed_statuses = range(200, 400)

    with requests.get(
        job_posting, timeout=30, allow_redirects=True, stream=True
    ) as resp:
        if resp.status_code in allowed_statuses:
            return job_posting
        else:
            raise RequestException("Job Posting could not be reached")


def handle_cli() -> argparse.Namespace:
    """Parse and validate CLI arguments for job application generator."""
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
