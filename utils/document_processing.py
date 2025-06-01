"""
Document processing utilities for parsing resumes and job descriptions.
"""

import logging
import os
import re
import json

from pathlib import Path
from urllib.parse import urlparse
from typing_extensions import Dict, List, Any


# Langchain imports
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers.json import JsonOutputParser
from langfuse.decorators import observe, langfuse_context
from pydantic import BaseModel, Field

# Local imports - using relative imports
from .errors import URLExtractionError, LLMProcessingError, JobDescriptionParsingError
from .llm_client import LLMClient
from ..prompts.templates import JOB_DESCRIPTION_PROMPT

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Default paths
DEFAULT_RESUME_PATH: str = os.getenv("DEFAULT_RESUME_PATH", "")


# Most Occurring Resume Section Headers
RESUME_SECTIONS: list[str] = [
    "EDUCATION", "EXPERIENCE", "SKILLS", "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE", "PROJECTS", "CERTIFICATIONS",
    "SUMMARY", "OBJECTIVE", "CONTACT", "PUBLICATIONS",
    "AWARDS", "LANGUAGES", "INTERESTS", "REFERENCES"
]

# Initialize LLM client
LLM: LLMClient = LLMClient()

llm_client: LLMClient = LLM.get_instance(
                            model_name="ejschwar/llama3.2-better-prompts:latest",
                            model_provider="ollama_json")
llm_structured = llm_client.get_llm()


class ResumeSection(BaseModel):
    """Model for a structured resume section."""
    title: str = Field(description="The section title (e.g., 'Experience', 'Education')")
    content: str = Field(description="The full content of this section")


class StructuredResume(BaseModel):
    """Model for a structured resume with sections."""
    sections: List[ResumeSection] = Field(description="List of resume sections")
    contact_info: Dict[str, str] = Field(description="Contact information extracted from the resume")

class JobDescriptionComponents(BaseModel):
    """Model for job description components."""
    company_name: str = Field(description="The company name")
    job_description: str = Field(description="The job description")
    reasoning: str = Field(description="The reasoning for the extracted information")

@observe()
def clean_resume_text(text: str) -> str:
    """Clean and normalize resume text by removing extra whitespace, fixing common PDF extraction issues.

    Args:
        text: Raw text extracted from resume

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text)  # Fix hyphenated words

    # Remove header/footer page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Replace bullet variations with standard markdown bullets
    text = re.sub(r'[•●○◘◙♦♣♠★]', '* ', text)

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
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        contact_info['email'] = email_match.group(0)

    # Extract phone (various formats)
    phone_match = re.search(r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        contact_info['phone'] = phone_match.group(0)

    # Extract LinkedIn URL
    linkedin_match = re.search(r'linkedin\.com/in/[a-zA-Z0-9_-]+/?', text)
    if linkedin_match:
        contact_info['linkedin'] = 'https://www.' + linkedin_match.group(0)

    # Try to extract name (this is approximate and might need LLM for better accuracy)
    # Typically name appears at the top of the resume
    first_line = text.strip().split('\n')[0].strip()
    if len(first_line) < 40 and not any(char.isdigit() for char in first_line):
        contact_info['name'] = first_line

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
    section_pattern = r'(?:^|\n)(?:[^a-zA-Z\d\s]|\s)*(' + '|'.join(RESUME_SECTIONS) + r')(?:[^a-zA-Z\d\s]|\s)*(?:$|\n)'
    matches = list(re.finditer(section_pattern, text, re.IGNORECASE))

    if not matches:
        # If no sections found, treat the whole resume as one section
        sections.append({
            "title": "resume",
            "content": text,
        })
        return sections

    # Process each section
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_pos = match.start()

        # Find the end position (start of next section or end of text)
        end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)

        # Extract section content (excluding the header)
        section_content = text[start_pos:end_pos].strip()

        sections.append({
            "title": section_title.lower(),
            "content": section_content
        })

    return sections


def _collapse_ws(text: str) -> str:
    """Collapse stray whitespace but keep bullet breaks."""
    text = re.sub(r"\n\s*([•\-–])\s*", r"\n\1 ", text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).replace(" \n", "\n").strip()


def _is_heading(line: str) -> bool:
    return (
        line.isupper()
        and len(line.split()) <= 5
        and not re.search(r"\d", line)
    )

def parse_resume(file_path: str | Path) -> List[Document]:
    """
    Load a résumé from PDF or TXT file → list[Document] chunks
    (≈400 chars, 50‑char overlap) with {source, section} metadata.
    """
    file_extension = Path(file_path).suffix.lower()
    
    # Handle different file types
    if file_extension == '.pdf':
        text = PyPDFLoader(str(file_path), extraction_mode="layout").load()[0].page_content
    elif file_extension == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if not text.strip():
                    raise ValueError("File is empty")
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise ValueError(f"Could not read text file: {file_path}. Error: {str(e)}")
    else:
        raise ValueError(f"Unsupported resume file type: {file_path}. Supported types: .pdf, .txt")
        
    text = _collapse_ws(text)

    # Tag headings with "###" so Markdown splitter can see them
    tagged_lines = [
        f"### {ln}" if _is_heading(ln) else ln
        for ln in text.splitlines()]
    
    md_text = "\n".join(tagged_lines)

    if "###" in md_text:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("###", "section")]
        )
        chunks = splitter.split_text(md_text)  # already returns Documents
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50
        )
        chunks: list[Document] = [Document(page_content=chunk, metadata={}) for chunk in splitter.split_text(md_text)]    # Attach metadata
    for doc in chunks:
        doc.metadata.setdefault("source", str(file_path))
        # section already present if header‑splitter was used

    return chunks


def get_job_description(file_path_or_url: str) -> Document:
    """Parse a job description from a file or URL into chunks.

    Args:
        file_path_or_url: Local file path or URL of job posting

    Returns:

        Document containing the job description
    """
    # Check if the input is a URL
    if file_path_or_url.startswith(('http://', 'https://')):
        return parse_job_desc_from_url(file_path_or_url)

    # Handle local files based on extension
    file_extension = Path(file_path_or_url).suffix.lower()
    
    # Handle txt files
    if file_extension == '.txt':
        try:
            with open(file_path_or_url, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError(f"File is empty: {file_path_or_url}")
                return Document(page_content=content, metadata={"source": file_path_or_url})
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise ValueError(f"Could not read text file: {file_path_or_url}. Error: {str(e)}")
    
    # For other file types
    raise ValueError(f"Unsupported file type: {file_path_or_url}. Supported types: .pdf, .docx, .txt, .md")


def parse_job_desc_from_url(url: str) -> Document:
    """Extract job description from a URL.

    Args:
        url: URL of the job posting

    Returns:
        List[str]: [job_description_markdown, company_name]

    Raises:
        ValueError: If URL format is invalid
        URLExtractionError: If content extraction fails
        LLMProcessingError: If LLM processing fails
    """
    
    logger.info("Starting job description extraction from URL: %s", url)
    # langfuse_handler = langfuse_context.get_current_langchain_handler()
    extracted_text = None
    
    try:
        # Validate URL format
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            logger.error("Invalid URL format: %s", url)
            raise ValueError("URL must start with http:// or https://")

        # Extract content from URL
        try:
            loader = WebBaseLoader(url)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            document_splitted = loader.load_and_split(text_splitter=text_splitter)
            
            if not document_splitted:
                logger.error("No content could be extracted from URL: %s", url)
                raise URLExtractionError("No content could be extracted from URL")
            
            extracted_text = " ".join(doc.page_content for doc in document_splitted)
            logger.info("Successfully extracted %d characters from URL", len(extracted_text))
            
        except Exception as e:
            raise URLExtractionError(f"Failed to extract content from URL: {str(e)}") from e

        # Process with LLM
        if not llm_structured:
            logger.warning("LLM not available, returning raw extracted text")
            return [extracted_text, "Unknown Company"]

        try:
            output_parser: JsonOutputParser = JsonOutputParser(pydantic_object=JobDescriptionComponents)

            human_prompt = "Below is the job description enclosed in triple quotes:\n\n '''{extracted_text}'''\n\n"
            
            job_description_parser_system_message = SystemMessagePromptTemplate.from_template(
                                                    template=JOB_DESCRIPTION_PROMPT,
                                                    input_variables=[])
            job_description_parser_human_message = HumanMessagePromptTemplate.from_template(
                                                    template=human_prompt,
                                                    input_variables=["extracted_text"])
            chat_prompt = ChatPromptTemplate.from_messages([job_description_parser_system_message, job_description_parser_human_message])

            # print("Chat prompt created successfully")
            chain = chat_prompt | llm_structured | output_parser
            
            try:
                # Process with LLM

                try:
                    result = chain.invoke({"extracted_text": extracted_text})
                except Exception as e:
                    logger.error("LLM invocation failed: %s", str(e))
                    raise LLMProcessingError(f"LLM invocation failed: {str(e)}") from e
                print("LLM processing result: ", result)
                # Handle different types of LLM results
                if isinstance(result, JobDescriptionComponents):
                    # Direct Pydantic model
                    result = result.model_dump()
                if isinstance(result, dict):
                    print("LLM returned a dictionary, converting to JobDescriptionComponents model", result)
                else:
                    # Unexpected result type
                    print(f"Unexpected LLM result type: {type(result)}")
                    logger.error("Unexpected LLM result type: %s", type(result))
                    raise LLMProcessingError("Invalid LLM response format")

                # Validate required fields
                if not result.get("job_description") or not result.get("company_name"):
                    logger.warning("LLM returned empty required fields")
                    raise LLMProcessingError("Missing required fields in LLM response")

                logger.info("Successfully processed job description with LLM")
                # Create a Document object for the job description
                job_doc = Document(
                    page_content=result["job_description"],
                    metadata={"company_name": result["company_name"]}
                )

                # print("Job description Document created successfully. Company name: ", result["company_name"])
                # print("Job description content: ", job_doc.metadata)  # Print first 100 chars for debugging
                return job_doc
            
            except Exception as e:
                # Handle LLM processing errors first
                if isinstance(e, LLMProcessingError):
                    raise
                
                # Try to recover from JSON parsing errors
                error_msg = str(e)
                if "Invalid json output" in error_msg:
                    logger.warning("Attempting to recover from invalid JSON output")
                    
                    # Extract JSON from error message
                    output = error_msg.split("Invalid json output:", 1)[1].strip()
                    start = output.find('{')
                    end = output.rfind('}') + 1
                    
                    if start >= 0 and end > start:
                        try:
                            clean_json = output[start:end]
                            result = output_parser.parse(clean_json)
                            if hasattr(result, "job_description") and hasattr(result, "company_name"):
                                return [result.job_description, result.company_name]
                        except json.JSONDecodeError as json_e:
                            logger.error("Failed to recover from JSON error: %s", json_e)
                
                raise LLMProcessingError(f"Failed to process job description with LLM: {str(e)}") from e

        except Exception as e:
            if isinstance(e, LLMProcessingError):
                if extracted_text:
                    logger.warning("LLM processing failed, falling back to raw text")
                    raise e
                    return [extracted_text, "Unknown Company"]
            raise LLMProcessingError(f"Failed to process job description with LLM: {str(e)}") from e

    except ValueError as e:
        logger.error("URL validation error: %s", str(e))
        raise
    except URLExtractionError as e:
        logger.error("Content extraction error: %s", str(e))
        raise
    except LLMProcessingError as e:
        if extracted_text:
            logger.warning("Using extracted text as fallback")
            return [extracted_text, "Unknown Company"]
        raise
    except Exception as e:
        logger.error("Unexpected error during job description parsing: %s", str(e))
        raise JobDescriptionParsingError(f"Failed to parse job description: {str(e)}") from e