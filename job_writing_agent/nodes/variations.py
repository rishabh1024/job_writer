import logging
from datetime import datetime
from typing_extensions import Dict, List

from langchain_core.documents import Document


from ..classes.classes import AppState
from ..utils.llm_provider_factory import LLMFactory
from ..prompts.templates import VARIATION_PROMPT


logger = logging.getLogger(__name__)
# Constants
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")

llm_provider = LLMFactory()

llm = llm_provider.create_langchain(
    "qwen/qwen3-4b:free", provider="openrouter", temperature=0.3
)


def generate_variations(state: AppState) -> Dict[str, List[str]]:
    """Generate multiple variations of the draft for self-consistency voting."""
    variations = []

    # Get resume and job text, handling both string and Document types
    try:
        resume_text = "\n".join(
            doc.page_content if isinstance(doc, Document) else doc
            for doc in (
                state["resume"][:2]
                if isinstance(state["company_research_data"]["resume"], str)
                else [state["resume"]]
            )
        )
        job_text = "\n".join(
            chunk
            for chunk in (
                state["company_research_data"]["job_description"][:2]
                if isinstance(state["company_research_data"]["job_description"], str)
                else [state["company_research_data"]["job_description"]]
            )
        )
    except Exception as e:
        print(f"Warning: Error processing resume/job text: {e}")
        # Fallback to simple string handling
        resume_text = str(state["company_research_data"]["resume"])
        job_text = str(state["company_research_data"]["job_description"])

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
                resume_excerpt=resume_text, job_excerpt=job_text, draft=state["draft"]
            )

            response = configured_llm.invoke(variation)

            if response and response.strip():  # Only add non-empty variations
                variations.append(response)
        except Exception as e:
            print(f"Warning: Error generating variation with settings {settings}: {e}")
            continue

    # Ensure we have at least one variation
    if not variations:
        # If all variations failed, add the original draft as a fallback
        variations.append(state["draft"])

    return {"variations": variations}
