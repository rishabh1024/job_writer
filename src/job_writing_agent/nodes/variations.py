# Standard library imports
import logging
from datetime import datetime

# Third-party imports
from langchain_core.documents import Document

# Local imports
from ..classes.classes import ResultState
from ..prompts.templates import VARIATION_PROMPT
from ..utils.llm_provider_factory import LLMFactory

logger = logging.getLogger(__name__)
# Constants
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")


def generate_variations(state: ResultState) -> dict[str, list[str]]:
    """
    Generate multiple variations of the draft for self-consistency voting.

    Args:
        state: Current result state with draft and research data

    Returns:
        Dictionary containing list of draft variations
    """
    # Validate and extract all required state fields once
    company_research_data = state.get("company_research_data", {})
    draft_content = state.get("draft", "")
    resume_data = company_research_data.get("resume", "")
    job_description_data = company_research_data.get("job_description", "")

    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm = llm_provider.create_langchain(
        "google/gemma-3-27b-it:free", provider="openrouter", temperature=0.3
    )

    variations = []

    # Get resume and job text, handling both string and Document types
    try:
        # Extract resume text
        if isinstance(resume_data, str):
            resume_text = resume_data[:2000]  # Limit to first 2000 chars
        elif isinstance(resume_data, list):
            resume_text = "\n".join(
                doc.page_content if isinstance(doc, Document) else str(doc)
                for doc in resume_data[:2]
            )
        else:
            resume_text = str(resume_data)

        # Extract job description text
        if isinstance(job_description_data, str):
            job_text = job_description_data[:2000]  # Limit to first 2000 chars
        elif isinstance(job_description_data, list):
            job_text = "\n".join(str(chunk) for chunk in job_description_data[:2])
        else:
            job_text = str(job_description_data)

    except Exception as e:
        logger.warning(f"Error processing resume/job text: {e}")
        # Fallback to simple string handling
        resume_text = str(resume_data)
        job_text = str(job_description_data)

    # Generate variations with different temperatures and creativity settings
    temp_variations = [
        {"temperature": 0.7, "top_p": 0.9},  # More conservative
        {"temperature": 0.75, "top_p": 0.92},  # Balanced
        {"temperature": 0.8, "top_p": 0.95},  # More creative
        {"temperature": 0.7, "top_p": 0.85},  # Alternative conservative
        {"temperature": 0.8, "top_p": 0.98},  # Most creative
    ]

    for settings in temp_variations:
        try:
            # Create a configured version of the LLM with the variation settings
            configured_llm = llm.with_config(configurable=settings)

            # Use VARIATION_PROMPT directly with the configured LLM
            variation = VARIATION_PROMPT.format_messages(
                resume_excerpt=resume_text, job_excerpt=job_text, draft=draft_content
            )

            response = configured_llm.invoke(variation)

            logger.debug(f"Generated variation with settings {settings}")

            if response and response.strip():  # Only add non-empty variations
                variations.append(response)
        except Exception as e:
            logger.warning(f"Error generating variation with settings {settings}: {e}")
            continue

    # Ensure we have at least one variation
    if not variations:
        # If all variations failed, add the original draft as a fallback
        logger.warning("All variations failed, using original draft as fallback")
        variations.append(draft_content)

    return {"variations": variations}
