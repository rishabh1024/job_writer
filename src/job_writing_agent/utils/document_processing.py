"""
Document processing utilities for parsing resumes and job descriptions.
"""

# Standard library imports
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

# Third-party imports
import dspy
from langchain_community.document_loaders import PyPDFLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langfuse import observe
from pydantic import BaseModel, Field
from typing_extensions import Any

# Local imports
from .errors import JobDescriptionParsingError, LLMProcessingError, URLExtractionError
from job_writing_agent.agents.output_schema import CandidateJobFitAnalysis

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
    reasoning: str = Field(
        description="The reasoning for the extracted information"
    )


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
    location = dspy.OutputField(
        desc="The location for the provided job posting."
    )


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

        sections.append(
            {"title": section_title.lower(), "content": section_content}
        )

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


def parse_resume(file_path: str | Path) -> str:
    """
    Load a résumé from PDF or TXT and return full text with structure preserved.

    Uses PyPDFLoader with extraction_mode="layout" for PDFs so layout and
    structure are preserved. All pages are concatenated. No chunking is applied.

    Parameters
    ----------
    file_path : str | Path
        Local path to a .pdf or .txt resume file.

    Returns
    -------
    str
        Full resume text (whitespace normalized via _collapse_ws).

    Raises
    ------
    ValueError
        Unsupported type, empty file, or read error.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path), extraction_mode="layout")
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs) if docs else ""
        if not text.strip():
            raise ValueError("PDF produced no text")
    elif suffix == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                if not text.strip():
                    raise ValueError("File is empty")
        except OSError as e:
            logger.error("Error reading text file: %s", e)
            raise ValueError(f"Could not read text file: {path}. Error: {e}") from e
    else:
        raise ValueError(
            f"Unsupported resume file type: {path}. Supported types: .pdf, .txt"
        )

    return _collapse_ws(text)


async def parse_job_description(url: str) -> Document:
    """Parse a job description from a URL. Validates URL then fetches and structures content.

    Only URLs are supported. Validation is done here; parse_job_description_from_url
    assumes the URL is already validated.

    Args:
        url: URL of the job posting (http:// or https://).

    Returns:
        Document containing the job description and metadata (company_name, etc.).

    Raises:
        ValueError: If the URL format is invalid.
    """
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        logger.error("Invalid URL format: %s", url)
        raise ValueError("URL must be valid and start with http:// or https://")
    return await parse_job_description_from_url(url)


async def scrape_job_description_from_web(urls: list[str]) -> str:
    """Scrape job listing URL(s) and return full page content as markdown text.

    Uses headless Chromium to load the page(s), then Html2TextTransformer to
    convert HTML to readable markdown. Returns one string (no chunking).
    """
    loader = AsyncChromiumLoader(urls, headless=True)
    scraped_docs = await loader.aload()

    html2text = Html2TextTransformer()
    markdown_docs = html2text.transform_documents(scraped_docs)

    return "\n".join(doc.page_content for doc in markdown_docs if doc.page_content)


async def parse_job_description_from_url(url: str) -> Document:
    """Extracts and structures a job description from a URL using an LLM.

    Caller (parse_job_description) is responsible for URL validation. This function
    fetches content, uses DSPy to extract key details, and returns a structured
    LangChain Document. If the LLM processing fails, it falls back to raw extracted text.

    Args:
        url: The URL of the job posting (assumed already validated).

    Returns:
        A Document containing the structured job description and company name in metadata.

    Raises:
        JobDescriptionParsingError: For any unexpected errors during the process.
    """
    logger.info("Starting job description extraction from URL: %s", url)

    raw_content = None
    try:
        # 2. Fetch content from the URL
        try:
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
                raise ValueError(
                    "CEREBRAS_API_KEY environment variable not set"
                )

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
                result = job_extract_fn(
                    job_description_html_content=raw_content
                )
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
            raise LLMProcessingError(
                f"Failed to process content with LLM: {e}"
            ) from e

    # 5. Handle specific, known errors
    except LLMProcessingError as e:
        logger.warning(f"LLM processing failed: {e}. Falling back to raw text.")
        # This is the corrected fallback logic. It uses the fetched `raw_content`.
        if raw_content:
            return Document(
                page_content=raw_content,
                metadata={
                    "company_name": "Unknown",
                    "source": url,
                    "error": str(e),
                },
            )
        # If raw_content is also None, then the failure was catastrophic.
        raise LLMProcessingError(
            "LLM processing failed and no raw content was available for fallback."
        ) from e

    except URLExtractionError as e:
        logger.error(f"Could not extract content from URL: {e}")
        raise URLExtractionError(
            "Failed to extract content from the URL."
        ) from e

    # 6. Catch any other unexpected errors
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise JobDescriptionParsingError(
            f"An unexpected error occurred while parsing the job description: {e}"
        ) from e


async def analyze_candidate_job_fit(
    resume_text: str,
    job_description: str,
    company_name: str,
) -> dict[str, Any]:
    """
    Analyze candidate-job fit using DSPy.

    Takes the candidate's resume and job description and produces structured
    analysis for downstream content generation (cover letter, bullets, etc.).

    Parameters
    ----------
    resume_text : str
        Full text of the candidate's resume.
    job_description : str
        Full text of the job posting.
    company_name : str
        Name of the company (can be empty).

    Returns
    -------
    dict[str, Any]
        Dictionary with analysis fields: matching_qualifications, transferable_skills,
        experience_highlights, potential_gaps, unique_value_proposition, talking_points.

    Raises
    ------
    ValueError
        If CEREBRAS_API_KEY environment variable is not set.
    """

    cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
    if not cerebras_api_key:
        raise ValueError("CEREBRAS_API_KEY environment variable not set")

    logger.info("Starting candidate-job fit analysis...")

    with dspy.context(
        lm=dspy.LM(
            "cerebras/qwen-3-32b",
            api_key=cerebras_api_key,
            temperature=0.2,
        )
    ):
        analyzer = dspy.Predict(CandidateJobFitAnalysis)
        result = analyzer(
            resume_text=resume_text,
            job_description=job_description,
            company_name=company_name or "the company",
        )

    logger.info("Candidate-job fit analysis completed.")

    return {
        "matching_qualifications": result.matching_qualifications,
        "transferable_skills": result.transferable_skills,
        "experience_highlights": result.experience_highlights,
        "potential_gaps": result.potential_gaps,
        "unique_value_proposition": result.unique_value_proposition,
        "talking_points": result.talking_points,
    }
