"""
Document processing utilities for parsing resumes and job descriptions.
"""

# Standard library imports
import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Third-party imports
import dspy
import httpx
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langfuse import observe
from pydantic import BaseModel, Field
from typing_extensions import Any

# Local imports
from .errors import (
    JobDescriptionParsingError,
    LLMProcessingError,
    ResumeDownloadError,
    URLExtractionError,
)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default paths
DEFAULT_RESUME_PATH: str = os.getenv("DEFAULT_RESUME_PATH", "")


# Most Occurring Resume Section Headers
RESUME_SECTIONS: list[str] = [
    "EDUCATION",
    "EXPERIENCE",
    "SKILLS",
    "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "PROJECTS",
    "CERTIFICATIONS",
    "SUMMARY",
    "OBJECTIVE",
    "CONTACT",
    "PUBLICATIONS",
    "AWARDS",
    "LANGUAGES",
    "INTERESTS",
    "REFERENCES",
]


class ResumeSection(BaseModel):
    """Model for a structured resume section."""

    title: str = Field(
        description="The section title (e.g., 'Experience', 'Education')"
    )
    content: str = Field(description="The full content of this section")


class StructuredResume(BaseModel):
    """Model for a structured resume with sections."""

    sections: list[ResumeSection] = Field(description="List of resume sections")
    contact_info: dict[str, str] = Field(
        description="Contact information extracted from the resume"
    )


class JobDescriptionComponents(BaseModel):
    """Model for job description components."""

    company_name: str = Field(description="The company name")
    job_description: str = Field(description="The job description")
    reasoning: str = Field(description="The reasoning for the extracted information")


class ExtractJobDescription(dspy.Signature):
    """Clean and extract the job description from the provided scraped HTML of the job posting.
    Divide the job description into multiple sections under different headings.Company Overview,
    Role Introduction,Qualifications and Requirements, Prefrred Qualifications, Salary, Location.
    Do not alter the content of the job description.
    """

    job_description_html_content = dspy.InputField(
        desc="HTML content of the job posting."
    )
    job_description = dspy.OutputField(
        desc="Clean job description which is free of HTML tags and irrelevant information."
    )
    job_role = dspy.OutputField(desc="The job role in the posting.")
    company_name = dspy.OutputField(desc="Company Name of the Job listing.")
    location = dspy.OutputField(desc="The location for the provided job posting.")


@observe()
def clean_resume_text(text: str) -> str:
    """Clean and normalize resume text by removing extra whitespace, fixing common PDF extraction issues.

    Args:
        text: Raw text extracted from resume

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix common PDF extraction issues
    text = re.sub(r"([a-z])- ([a-z])", r"\1\2", text)  # Fix hyphenated words

    # Remove header/footer page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Replace bullet variations with standard markdown bullets
    text = re.sub(r"[•●○◘◙♦♣♠★]", "* ", text)

    return text.strip()


@observe()
def extract_contact_info(text: str) -> dict[str, str]:
    """Extract contact information from resume text.

    Args:
        text: Resume text to extract from

    Returns:
        Dictionary with contact information
    """
    contact_info = {}

    # Extract email
    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
    )
    if email_match:
        contact_info["email"] = email_match.group(0)

    # Extract phone (various formats)
    phone_match = re.search(
        r"(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", text
    )
    if phone_match:
        contact_info["phone"] = phone_match.group(0)

    # Extract LinkedIn URL
    linkedin_match = re.search(r"linkedin\.com/in/[a-zA-Z0-9_-]+/?", text)
    if linkedin_match:
        contact_info["linkedin"] = "https://www." + linkedin_match.group(0)

    # Try to extract name (this is approximate and might need LLM for better accuracy)
    # Typically name appears at the top of the resume
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) < 40 and not any(char.isdigit() for char in first_line):
        contact_info["name"] = first_line

    return contact_info


@observe()
def identify_resume_sections(text: str) -> list[dict[str, Any]]:
    """Identify sections in a resume text.

    Args:
        text: Full resume text
        llm: Optional language model for advanced section detection

    Returns:
        List of dictionaries with section info
    """
    sections = []

    # if llm:
    #     # Use LLM for more accurate section identification
    #     prompt = ChatPromptTemplate.from_messages([
    #         SystemMessage(content="""You are an expert at parsing resumes.
    #         Identify the main sections in this resume text and structure them.
    #         For each section, extract the title and content."""),
    #         HumanMessage(content=f"Resume text:\n\n{text}")
    #     ])

    #     class ResumeStructure(BaseModel):
    #         sections: List[Dict[str, str]] = Field(description="List of identified sections with title and content")

    #     parser = PydanticOutputParser(pydantic_object=ResumeStructure)
    #     chain = prompt | llm | parser

    #     try:
    #         result = chain.invoke({})
    #         return result.sections
    #     except Exception as e:
    #         print(f"LLM section extraction failed: {e}")

    # Regex-based section identification
    # Create a pattern that matches common section headers
    section_pattern = (
        r"(?:^|\n)(?:[^a-zA-Z\d\s]|\s)*("
        + "|".join(RESUME_SECTIONS)
        + r")(?:[^a-zA-Z\d\s]|\s)*(?:$|\n)"
    )
    matches = list(re.finditer(section_pattern, text, re.IGNORECASE))

    if not matches:
        # If no sections found, treat the whole resume as one section
        sections.append(
            {
                "title": "resume",
                "content": text,
            }
        )
        return sections

    # Process each section
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_pos = match.start()

        # Find the end position (start of next section or end of text)
        end_pos = matches[i + 1].start() if i < len(matches) - 1 else len(text)

        # Extract section content (excluding the header)
        section_content = text[start_pos:end_pos].strip()

        sections.append({"title": section_title.lower(), "content": section_content})

    return sections


def _collapse_ws(text: str) -> str:
    """
    Collapse stray whitespace but keep bullet breaks.

    Args:
        text: Input text with potential whitespace issues

    Returns:
        Text with collapsed whitespace
    """
    text = re.sub(r"\n\s*([•\-–])\s*", r"\n\1 ", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).replace(" \n", "\n").strip()


def _is_heading(line: str) -> bool:
    """
    Check if a line is a heading (all uppercase, short, no digits).

    Args:
        line: Line of text to check

    Returns:
        True if line appears to be a heading
    """
    return line.isupper() and len(line.split()) <= 5 and not re.search(r"\d", line)


def _is_huggingface_hub_url(url: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if URL or string is a HuggingFace Hub reference and extract repo_id and filename.

    Args:
        url: URL or string to check (e.g., "https://huggingface.co/datasets/username/dataset/resolve/main/file.pdf"
            or "username/dataset-name::resume.pdf")

    Returns:
        Tuple of (is_hf_url, repo_id, filename). Returns (False, None, None) if not HF Hub.
    """
    if not url or not isinstance(url, str):
        return (False, None, None)

    # Custom format: "username/dataset-name::filename"
    if "::" in url and not url.startswith(("http://", "https://")):
        parts = url.split("::", 1)
        if len(parts) == 2 and "/" in parts[0] and parts[1].strip():
            return (True, parts[0].strip(), parts[1].strip())
        return (False, None, None)

    # HF Hub URL patterns
    if not url.startswith(("http://", "https://")):
        return (False, None, None)

    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return (False, None, None)

    # Pattern: /datasets/{username}/{dataset}/resolve/main/{filename}
    # Pattern: /datasets/{username}/{dataset}/blob/main/{filename}
    # Pattern: /{username}/{dataset}/resolve/main/{filename} (models)
    match = re.match(
        r"^/(?:datasets/)?([^/]+)/([^/]+)/(?:resolve|blob)/[^/]+/(.+)$",
        parsed.path,
    )
    if match:
        repo_id = f"{match.group(1)}/{match.group(2)}"
        filename = match.group(3)
        return (True, repo_id, filename)

    return (False, None, None)


async def download_file_from_hf_hub(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Download a file from HuggingFace Hub dataset or repository.

    Uses the huggingface_hub library with authentication and caching support.

    Args:
        repo_id: HF Hub repository ID (e.g., "username/dataset-name").
        filename: Name of the file to download (e.g., "resume.pdf").
        repo_type: Type of repository ("dataset" or "model"). Defaults to "dataset".
        token: Optional HF API token. If None, uses HUGGINGFACE_API_KEY env var.
        cache_dir: Optional cache directory. Defaults to HF_HOME env var or system temp.

    Returns:
        Path to the downloaded file (from cache or new download).

    Raises:
        ValueError: If repo_id or filename is invalid.
        ResumeDownloadError: If download fails.
    """
    if not repo_id or not isinstance(repo_id, str) or "/" not in repo_id:
        raise ValueError(
            f"Invalid repo_id: {repo_id}. Expected format: username/dataset-name"
        )
    if not filename or not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")

    hf_token = token or os.getenv("HUGGINGFACE_API_KEY")
    cache = (
        str(cache_dir) if cache_dir else os.getenv("HF_HOME") or tempfile.gettempdir()
    )

    def _download() -> str:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename.strip(),
            repo_type=repo_type,
            token=hf_token,
            cache_dir=cache,
        )

    try:
        logger.info("Downloading %s from HF Hub repo %s", filename, repo_id)
        local_path = await asyncio.to_thread(_download)
        logger.info("Downloaded resume to %s", local_path)
        return Path(local_path)
    except Exception as e:
        logger.error("Failed to download from HF Hub: %s", e)
        raise ResumeDownloadError(
            f"Could not download {filename} from {repo_id}: {e}"
        ) from e


async def download_file_from_url(
    url: str,
    save_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Download a file from an HTTP/HTTPS URL to a local temporary location.

    Handles generic web URLs (GitHub raw files, public cloud storage, etc.).
    For HuggingFace Hub, use download_file_from_hf_hub() instead.

    Args:
        url: The URL to download from (must start with http:// or https://).
        save_dir: Optional directory to save file. Defaults to system temp directory.
        filename: Optional filename. If not provided, inferred from URL or uses temp name.

    Returns:
        Path to the downloaded file.

    Raises:
        ValueError: If URL format is invalid.
        ResumeDownloadError: If download fails.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc or parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    save_dir = save_dir or Path(tempfile.gettempdir())
    save_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = Path(parsed.path).name or "resume.pdf"

    local_path = save_dir / filename
    logger.info("Downloading resume from URL: %s", url)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            local_path.write_bytes(response.content)
        logger.info("Downloaded resume to %s", local_path)
        return local_path
    except httpx.HTTPError as e:
        logger.error("HTTP error downloading from %s: %s", url, e)
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        raise ResumeDownloadError(f"Could not download from {url}: {e}") from e
    except OSError as e:
        logger.error("Error writing file from %s: %s", url, e)
        raise ResumeDownloadError(f"Could not save file from {url}: {e}") from e


def parse_resume(file_path: str | Path) -> list[Document]:
    """
    Load a résumé from PDF or TXT file → list[Document] chunks
    (≈400 chars, 50‑char overlap) with {source, section} metadata.
    """
    file_extension = Path(file_path).suffix.lower()

    # Handle different file types
    if file_extension == ".pdf":
        text = (
            PyPDFLoader(str(file_path), extraction_mode="layout").load()[0].page_content
        )
    elif file_extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if not text.strip():
                    raise ValueError("File is empty")
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise ValueError(f"Could not read text file: {file_path}. Error: {str(e)}")
    else:
        raise ValueError(
            f"Unsupported resume file type: {file_path}. Supported types: .pdf, .txt"
        )

    text = _collapse_ws(text)

    # Tag headings with "###" so Markdown splitter can see them
    tagged_lines = [f"### {ln}" if _is_heading(ln) else ln for ln in text.splitlines()]

    md_text = "\n".join(tagged_lines)

    if "###" in md_text:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "section")])
        chunks = splitter.split_text(md_text)  # already returns Documents
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks: list[Document] = [
            Document(page_content=chunk, metadata={})
            for chunk in splitter.split_text(md_text)
        ]  # Attach metadata
    for doc in chunks:
        doc.metadata.setdefault("source", str(file_path))
        # section already present if header‑splitter was used
    return chunks


async def get_resume(file_path_or_url: str | Path) -> list[Document]:
    """
    Load a résumé from a local file path or URL.

    Handles both local files and URLs by downloading if needed, then delegating
    to parse_resume() for parsing. Supports HuggingFace Hub datasets and
    generic HTTP/HTTPS URLs.

    Args:
        file_path_or_url: Local file path, HF Hub reference, or URL.
            Examples:
            - Local: "/path/to/resume.pdf"
            - HF Hub URL: "https://huggingface.co/datasets/username/dataset/resolve/main/resume.pdf"
            - HF Hub format: "username/dataset-name::resume.pdf"
            - Generic HTTP: "https://example.com/resume.pdf"

    Returns:
        List of Document chunks with resume content.

    Raises:
        ResumeDownloadError: If URL download fails.
        ValueError: If file path is invalid or unsupported format.
    """
    source = str(file_path_or_url)

    # 1. Check if HuggingFace Hub URL or custom format
    is_hf, repo_id, filename = _is_huggingface_hub_url(source)
    if is_hf and repo_id and filename:
        local_path = await download_file_from_hf_hub(repo_id=repo_id, filename=filename)
        return parse_resume(local_path)

    # 2. Check if generic HTTP/HTTPS URL
    if source.startswith(("http://", "https://")):
        local_path = await download_file_from_url(source)
        return parse_resume(local_path)

    # 3. Treat as local file path
    return parse_resume(
        Path(source) if isinstance(file_path_or_url, str) else file_path_or_url
    )


async def get_job_description(file_path_or_url: str) -> Document:
    """Parse a job description from a file or URL into chunks.

    Args:
        file_path_or_url: Local file path or URL of job posting

    Returns:
        Document containing the job description
    """
    # Check if the input is a URL
    if file_path_or_url.startswith(("http://", "https://")):
        return await parse_job_description_from_url(file_path_or_url)

    # Handle local files based on extension
    file_extension = Path(file_path_or_url).suffix.lower()

    # Handle txt files
    if file_extension == ".txt":
        try:
            with open(file_path_or_url, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    raise ValueError(f"File is empty: {file_path_or_url}")
                return Document(
                    page_content=content, metadata={"source": file_path_or_url}
                )
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise ValueError(
                f"Could not read text file: {file_path_or_url}. Error: {str(e)}"
            )

    # For other file types
    raise ValueError(
        f"Unsupported file type: {file_path_or_url}. Supported types: .pdf, .docx, .txt, .md"
    )


async def scrape_job_description_from_web(urls: list[str]) -> str:
    """This function will first scrape the data from the job listing.
    Then using the recursive splitter using the different seperators,
    it preserves the paragraphs, lines and words"""
    loader = AsyncChromiumLoader(urls, headless=True)
    scraped_data_documents = await loader.aload()

    html2text = Html2TextTransformer()
    markdown_scraped_data_documents = html2text.transform_documents(
        scraped_data_documents
    )

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )

    extracted_content = splitter.split_documents(markdown_scraped_data_documents)

    return ".".join(doc.page_content for doc in extracted_content)


async def parse_job_description_from_url(url: str) -> Document:
    """Extracts and structures a job description from a URL using an LLM.

    This function fetches content from a URL, uses a DSPy to extract key details,
    and returns a structured LangChain Document. If the LLM processing fails, it falls
    back to returning the raw extracted text.

    Args:
        url: The URL of the job posting.

    Returns:
        A Document containing the structured job description and company name in metadata.

    Raises:
        ValueError: If the URL format is invalid.
        JobDescriptionParsingError: For any unexpected errors during the process.
    """
    logger.info("Starting job description extraction from URL: %s", url)

    # 1. Validate URL
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        logger.error("Invalid URL format: %s", url)
        raise ValueError("URL must be valid and start with http:// or https://")

    raw_content = None
    try:
        # 2. Fetch content from the URL
        try:
            logger.info("Fetching content from URL...")
            raw_content = await scrape_job_description_from_web([url])
            if not raw_content or not raw_content.strip():
                raise URLExtractionError(
                    "Failed to extract any meaningful content from the URL."
                )
            logger.info("Successfully fetched raw content from URL.")
        except Exception as e:
            # Wrap any fetching error into our custom exception
            raise URLExtractionError(
                f"Failed to download or read content from {url}: {e}"
            ) from e

        # 3. Process content with the LLM
        try:
            logger.info("Processing content with DSPy LLM...")
            # Configure DSPy LM with safe environment variable access
            cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
            if not cerebras_api_key:
                raise ValueError("CEREBRAS_API_KEY environment variable not set")

            # Use dspy.context() for async tasks instead of dspy.configure()
            with dspy.context(
                lm=dspy.LM(
                    "cerebras/qwen-3-32b",
                    api_key=cerebras_api_key,
                    temperature=0.1,
                    max_tokens=60000,  # Note: This max_tokens is unusually high
                )
            ):
                job_extract_fn = dspy.Predict(ExtractJobDescription)
                result = job_extract_fn(job_description_html_content=raw_content)
            logger.info("Successfully processed job description with LLM.")

            # 4. Create the final Document with structured data
            job_doc = Document(
                page_content=result.job_description,
                metadata={
                    "company_name": result.company_name,
                    "source": url,
                    "job_role": result.job_role,
                    "location": result.location,
                },
            )
            return job_doc

        except Exception as e:
            # Wrap any LLM error into our custom exception
            raise LLMProcessingError(f"Failed to process content with LLM: {e}") from e

    # 5. Handle specific, known errors
    except LLMProcessingError as e:
        logger.warning(f"LLM processing failed: {e}. Falling back to raw text.")
        # This is the corrected fallback logic. It uses the fetched `raw_content`.
        if raw_content:
            return Document(
                page_content=raw_content,
                metadata={"company_name": "Unknown", "source": url, "error": str(e)},
            )
        # If raw_content is also None, then the failure was catastrophic.
        raise LLMProcessingError(
            "LLM processing failed and no raw content was available for fallback."
        ) from e

    except URLExtractionError as e:
        logger.error(f"Could not extract content from URL: {e}")
        raise URLExtractionError("Failed to extract content from the URL.") from e

    # 6. Catch any other unexpected errors
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise JobDescriptionParsingError(
            f"An unexpected error occurred while parsing the job description: {e}"
        ) from e
