import logging
from datetime import datetime

from ..classes.classes import AppState
from ..prompts.templates import (
    DRAFT_RATING_PROMPT,
    BEST_DRAFT_SELECTION_PROMPT
)


logger = logging.getLogger(__name__)
# Constants
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")

# LLM = LLMClient()
# llm = LLMClient().get_llm()
# llm_precise = LLMClient().get_llm()


def self_consistency_vote(state: AppState) -> AppState:
    """Choose the best draft from multiple variations."""
    variations = state.get("variations", {"variations": []})

    all_drafts = [state["draft"]] + variations["variations"]

    # First, have the LLM rate each draft
    ratings = []

    # Get resume and job summaries, handling different formats
    try:
        if isinstance(state["resume"], list) and len(state["resume"]) > 0:
            if hasattr(state["resume"][0], 'page_content'):
                resume_summary = state["resume"][0].page_content
            else:
                resume_summary = state["resume"][0]
        else:
            resume_summary = str(state["resume"])
    except Exception as e:
        print(f"Warning: Error getting resume summary: {e}")
        resume_summary = str(state["resume"])

    try:
        if isinstance(state["job_description"], list) and len(state["job_description"]) > 0:
            job_summary = state["job_description"][0]
        else:
            job_summary = str(state["job_description"])
    except Exception as e:
        print(f"Warning: Error getting job summary: {e}")
        job_summary = str(state["job_description"])

    for i, draft in enumerate(all_drafts):
        rating = llm_precise.invoke(DRAFT_RATING_PROMPT.format(
            resume_summary=resume_summary,
            job_summary=job_summary,
            draft=draft,
            draft_number=i+1
        ))
        ratings.append(rating)

    # Create a clearer, more structured prompt for draft selection
    selection_prompt = BEST_DRAFT_SELECTION_PROMPT.format(
        ratings_json=json.dumps(ratings, indent=2),
        num_drafts=len(all_drafts)
    )

    # Get the selected draft index with error handling
    try:
        selection = llm_precise.invoke(selection_prompt).strip()
        # Extract just the first number found in the response
        number_match = re.search(r'\d+', selection)
        if not number_match:
            print("Warning: Could not extract draft number from LLM response. Using original draft.")
            best_draft_idx = 0
        else:
            best_draft_idx = int(number_match.group()) - 1
            # Validate the index is in range
            if best_draft_idx < 0 or best_draft_idx >= len(all_drafts):
                print(f"Warning: Selected draft index {best_draft_idx + 1} out of range. Using original draft.")
                best_draft_idx = 0
    except (ValueError, TypeError) as e:
        print(f"Warning: Error selecting best draft: {e}. Using original draft.")
        best_draft_idx = 0

    state["draft"] = all_drafts[best_draft_idx]
    return state
