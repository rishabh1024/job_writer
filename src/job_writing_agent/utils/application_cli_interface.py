import argparse
import socket
import tempfile
from pathlib import Path
from typing import Iterable
import re

import requests
from urllib3.exceptions import NameResolutionError


DEFAULT_MODEL = "allenai/olmo-3.1-32b-think:free"
DEFAULT_CONTENT_TYPE = "cover_letter"
SUPPORTED_FILE_EXTENSIONS = {".pdf", ".md", ".json", ".txt"}
VALID_CONTENT_TYPES = ["cover_letter", "bullets", "linkedin_note"]
DEFAULT_CONTENT_TYPE = "cover_letter"
DEFAULT_MODEL_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 30
TEMP_MIN, TEMP_MAX = 0.0, 2.0

# Google Docs patterns and export formats
GOOGLE_DOCS_PATTERN = r'https://docs\.google\.com/document/d/([a-zA-Z0-9-_]+)'
GOOGLE_DOCS_EXPORT_FORMATS = {
    'pdf': 'application/pdf',
    'txt': 'text/plain',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def is_google_docs_url(url: str) -> bool:
    """
    Check if the given URL is a Google Docs sharing link.
    
    Args:
        url: URL string to check
        
    Returns:
        True if it's a Google Docs URL, False otherwise
    """
    return bool(re.match(GOOGLE_DOCS_PATTERN, url))


def extract_google_docs_id(url: str) -> str | None:
    """
    Extract the document ID from a Google Docs URL.
    
    Args:
        url: Google Docs URL
        
    Returns:
        Document ID if found, None otherwise
    """
    match = re.search(GOOGLE_DOCS_PATTERN, url)
    return match.group(1) if match else None


def download_google_docs(url: str, export_format: str = 'txt') -> str:
    """
    Download a Google Docs document and save it to a temporary file.
    
    Args:
        url: Google Docs sharing URL
        export_format: Export format ('pdf', 'txt', 'docx')
        
    Returns:
        Path to downloaded temporary file
        
    Raises:
        ArgumentTypeError: If download fails or format is unsupported
    """
    doc_id = extract_google_docs_id(url)
    if not doc_id:
        raise argparse.ArgumentTypeError(f"Invalid Google Docs URL: {url}")
    
    if export_format not in GOOGLE_DOCS_EXPORT_FORMATS:
        raise argparse.ArgumentTypeError(
            f"Unsupported export format: {export_format}. "
            f"Supported formats: {list(GOOGLE_DOCS_EXPORT_FORMATS.keys())}"
        )
    
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format={export_format}"
    
    try:
        response = requests.get(export_url, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        
        # Create temporary file with appropriate extension
        suffix = f".{export_format}"
        with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
            
    except requests.exceptions.RequestException as e:
        raise argparse.ArgumentTypeError(
            f"Failed to download Google Docs document: {e}"
        )


def is_readable_file(path: str) -> str:
    """
    Validate that the file exists and has a supported extension, or download from Google Docs.
    Args:
        path: File path or Google Docs URL to validate
    Returns:
        Original path string if valid local file, or path to downloaded temp file for Google Docs
    Raises:
        ArgumentTypeError: If file doesn't exist, has unsupported extension, or download fails
    """
    # Check if it's a Google Docs URL
    if is_google_docs_url(path):
        # Try to download as text first (most compatible), fallback to PDF if needed
        try:
            return download_google_docs(path, 'txt')
        except argparse.ArgumentTypeError:
            # If text export fails, try PDF
            return download_google_docs(path, 'pdf')
    
    # Handle local file path
    file_path = Path(path)
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if not path.lower().endswith(tuple(SUPPORTED_FILE_EXTENSIONS)):
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
    if not (TEMP_MIN <= value <= TEMP_MAX):
        raise argparse.ArgumentTypeError(f"Temperature must be between {TEMP_MIN} and {TEMP_MAX}.")
    return value


def is_valid_url(job_posting: str, allowed_statuses: Iterable[int] | None = None) -> str:
    """Validate URL is reachable. Raises ArgumentTypeError if invalid."""
    if allowed_statuses is None:
        allowed_statuses = range(200, 400)

    try:
        response = requests.get(job_posting, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        if response.status_code not in allowed_statuses:
            raise argparse.ArgumentTypeError(f"URL returned status {response.status_code}")
        return job_posting
    except socket.gaierror as e:
        raise argparse.ArgumentTypeError(f"Domain name resolution failed: {e}")
    except requests.exceptions.ConnectionError as e:
        # Check if this ConnectionError was caused by a NameResolutionError
        if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
            raise argparse.ArgumentTypeError(f"ConnectionError. Domain name could not be resolved: {job_posting}")
        raise argparse.ArgumentTypeError(f"Connection failed: {e}")
    except requests.exceptions.Timeout as e:
        raise argparse.ArgumentTypeError(f"Request timed out: {e}")
    except requests.exceptions.InvalidURL as e:
        raise argparse.ArgumentTypeError(f"Invalid URL format: {e}")
    except requests.exceptions.RequestException as e:
        raise argparse.ArgumentTypeError(f"URL validation failed: {e}")


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
        type=is_readable_file,
        help="""
            Provide the path to the file containing the candidate's resume. \
            It can be a local file path or a Google Docs sharing URL.
            Supported formats are .pdf, .md, .txt, and .json.
            For Google Docs, the document will be downloaded automatically.
            """,
        )
    parser.add_argument(
        "-j",
        "--jd-source",
        required=True,
        metavar="jd_source",
        type=is_valid_url,
        help="URL to job posting or paste raw text of job description text.",
    )
    parser.add_argument(
        "-t",
        "--content_type",
        default=DEFAULT_CONTENT_TYPE,
        choices=VALID_CONTENT_TYPES,
        help=f"Type of application material to generate (default: {DEFAULT_CONTENT_TYPE}).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        metavar="model_nam",
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temp",
        type=valid_temp,
        default=DEFAULT_MODEL_TEMPERATURE,
        metavar="model_temperature",
        help=f"Temperature for the LLM, {TEMP_MIN}-{TEMP_MAX}.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    return parser.parse_args()
