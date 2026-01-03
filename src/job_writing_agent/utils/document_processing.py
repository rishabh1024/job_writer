"""
Document processing utilities for parsing resumes and job descriptions.
"""

import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from typing_extensions import Dict, List, Any


import dspy
from langchain_community.document_loaders import PyPDFLoader, AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from langfuse import observe
from pydantic import BaseModel, Field

# Local imports - using relative imports
from .errors import URLExtractionError, LLMProcessingError, JobDescriptionParsingError

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

    sections: List[ResumeSection] = Field(description="List of resume sections")
    contact_info: Dict[str, str] = Field(
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
def extract_contact_info(text: str) -> Dict[str, str]:
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
def identify_resume_sections(text: str) -> List[Dict[str, Any]]:
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
    """Collapse stray whitespace but keep bullet breaks."""
    text = re.sub(r"\n\s*([•\-–])\s*", r"\n\1 ", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).replace(" \n", "\n").strip()


def _is_heading(line: str) -> bool:
    return line.isupper() and len(line.split()) <= 5 and not re.search(r"\d", line)


def parse_resume(file_path: str | Path) -> List[Document]:
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


async def scrape_job_description_from_web(urls: List[str]):
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
            # Configure DSPy LM (it's good practice to do this here if it can change)
            dspy.configure(
                lm=dspy.LM(
                    "cerebras/qwen-3-32b",
                    api_key=os.environ.get("CEREBRAS_API_KEY"),
                    temperature=0.1,
                    max_tokens=60000,  # Note: This max_tokens is unusually high
                )
            )

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
