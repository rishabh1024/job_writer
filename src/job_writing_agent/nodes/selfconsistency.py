# Standard library imports
import json
import logging
import re
from datetime import datetime

# Local imports
from ..classes.classes import AppState
from ..prompts.templates import BEST_DRAFT_SELECTION_PROMPT, DRAFT_RATING_PROMPT
from ..utils.llm_provider_factory import LLMFactory


logger = logging.getLogger(__name__)
# Constants
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")


def self_consistency_vote(state: AppState) -> AppState:
    """
    Choose the best draft from multiple variations using LLM-based voting.

    This function rates all draft variations and selects the best one based on
    criteria like relevance, professional tone, personalization, and persuasiveness.

    Args:
        state: Application state containing the original draft and variations

    Returns:
        Updated state with the best draft selected
    """
    # Create LLM inside function (lazy initialization)
    llm_factory = LLMFactory()
    precise_llm = llm_factory.create_langchain(
        model="google/gemma-3-27b-it:free", provider="openrouter", temperature=0.1
    )

    variations_data = state.get("variations", {"variations": []})
    original_draft = state.get("draft", "")

    all_drafts = [original_draft] + variations_data.get("variations", [])

    # First, have the LLM rate each draft
    draft_ratings = []

    # Get resume and job summaries with safe dictionary access
    try:
        resume_path = state.get("resume_path", "")
        if isinstance(resume_path, list) and len(resume_path) > 0:
            if hasattr(resume_path[0], "page_content"):
                resume_summary = resume_path[0].page_content
            else:
                resume_summary = resume_path[0]
        else:
            resume_summary = str(resume_path)
    except Exception as e:
        logger.warning(f"Error getting resume summary: {e}")
        resume_summary = str(state.get("resume_path", ""))

    try:
        job_description_source = state.get("job_description_source", "")
        if isinstance(job_description_source, list) and len(job_description_source) > 0:
            job_summary = job_description_source[0]
        else:
            job_summary = str(job_description_source)
    except Exception as e:
        logger.warning(f"Error getting job summary: {e}")
        job_summary = str(state.get("job_description_source", ""))

    for draft_index, draft_content in enumerate(all_drafts):
        # Create chain with proper prompt template invocation
        rating_chain = DRAFT_RATING_PROMPT | precise_llm
        rating_result = rating_chain.invoke(
            {
                "resume_summary": resume_summary,
                "job_summary": job_summary,
                "draft": draft_content,
                "draft_number": draft_index + 1,
            }
        )
        draft_ratings.append(rating_result)

    # Create chain for draft selection with proper prompt template invocation
    selection_chain = BEST_DRAFT_SELECTION_PROMPT | precise_llm
    selection_result = selection_chain.invoke(
        {
            "ratings_json": json.dumps(draft_ratings, indent=2),
            "num_drafts": len(all_drafts),
        }
    )

    # Get the selected draft index with error handling
    try:
        selection_text = str(
            selection_result.content
            if hasattr(selection_result, "content")
            else selection_result
        ).strip()
        # Extract just the first number found in the response
        number_match = re.search(r"\d+", selection_text)
        if not number_match:
            logger.warning(
                "Could not extract draft number from LLM response. Using original draft."
            )
            best_draft_index = 0
        else:
            best_draft_index = int(number_match.group()) - 1
            # Validate the index is in range
            if best_draft_index < 0 or best_draft_index >= len(all_drafts):
                logger.warning(
                    f"Selected draft index {best_draft_index + 1} out of range. Using original draft."
                )
                best_draft_index = 0
    except (ValueError, TypeError) as e:
        logger.warning(f"Error selecting best draft: {e}. Using original draft.")
        best_draft_index = 0

    # Update state with best draft using safe dictionary operations
    updated_state = {**state, "draft": all_drafts[best_draft_index]}
    return updated_state
