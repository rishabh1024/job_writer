"""
Node functions for the job application writer LangGraph.

This module contains all the node functions used in the job application
writer workflow graph, each handling a specific step in the process.
"""

# Standard library imports
import logging
from datetime import datetime
from langgraph.types import interrupt

# Third-party imports
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# Local imports
from ..classes.classes import AppState, ResearchState, ResultState
from ..prompts.templates import (
    BULLET_POINTS_PROMPT,
    COVER_LETTER_PROMPT,
    DRAFT_GENERATION_CONTEXT_PROMPT,
    LINKEDIN_NOTE_PROMPT,
    REVISION_PROMPT,
)
from ..utils.llm_provider_factory import LLMFactory

logger = logging.getLogger(__name__)
# Constants
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")


def create_draft(state: ResearchState) -> ResultState:
    """Create initial draft of the application material."""
    # Validate state inputs
    company_background_information = state.get("company_research_data", {})
    if not company_background_information:
        logger.error("Missing company_research_data in state")
        raise ValueError("company_research_data is required in state")

    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm = llm_provider.create_langchain(
        "google/gemma-3-27b-it:free",
        provider="openrouter",
        temperature=0.3,
    )

    draft_category_map = {
        "cover_letter": COVER_LETTER_PROMPT,
        "bullets": BULLET_POINTS_PROMPT,
        "linkedin_connect_request": LINKEDIN_NOTE_PROMPT,
    }

    # Determine which type of content we're creating

    content_category = state.get("content_category", "cover_letter")

    # Select appropriate system message template based on content category
    logger.info(f"The candidate wants the Agent to assist with : {content_category}")
    system_message_template = draft_category_map.get(
        content_category, COVER_LETTER_PROMPT
    )

    # Build the complete prompt template: system message + context
    draft_prompt_template = ChatPromptTemplate([system_message_template])
    draft_prompt_template.append(DRAFT_GENERATION_CONTEXT_PROMPT)

    # Build the chain: input formatting -> prompt template -> LLM
    draft_generation_chain = (
        (
            {
                "current_job_role": lambda x: x["current_job_role"],
                "candidate_resume": lambda x: x["candidate_resume"],
                "company_research_data": lambda x: x["company_research_data"],
                "current_date": lambda x: x["current_date"],
            }
        )
        | draft_prompt_template
        | llm
    )

    # Prepare the inputs with safe dictionary access
    application_background_data = {
        "current_job_role": company_background_information.get("job_description", ""),
        "candidate_resume": company_background_information.get("resume", ""),
        "company_research_data": company_background_information.get(
            "company_research_data_summary", "Company Research Data is not available"
        ),
        "current_date": CURRENT_DATE,
    }

    response = draft_generation_chain.invoke(application_background_data)
    logger.info(f"Draft has been created: {response.content}")

    app_state = ResultState(
        draft=str(response.content),
        feedback="",
        critique_feedback="",
        current_node="create_draft",
        output_data="",
        company_research_data=state.get("company_research_data", {}),
        messages=state.get("messages", []),
    )

    return app_state


def critique_draft(state: ResultState) -> ResultState:
    """
    Critique the draft for improvements.
    Provides external evaluation focusing on job requirements, tone, clarity, and style.
    """
    try:
        logger.info("Critiquing draft...")

        # Validate and extract required state fields once at the start
        company_research_data = state.get("company_research_data", {})
        job_description = str(company_research_data.get("job_description", ""))
        draft_content = str(state.get("draft", ""))

        # Debug logging to verify values
        logger.debug(f"Job description length: {len(job_description)}")
        logger.debug(f"Draft length: {len(draft_content)}")

        # Early return if required fields are missing
        if not job_description or not draft_content:
            logger.warning("Missing content for critique in state")
            return ResultState(
                draft=state.get("draft", ""),
                feedback=state.get("feedback", ""),
                critique_feedback="",
                current_node="critique",
                output_data="",
                company_research_data=state.get("company_research_data", {}),
                messages=state.get("messages", []),
            )

        # Create LLM inside function (lazy initialization)
        llm_provider = LLMFactory()
        llm = llm_provider.create_langchain(
            "google/gemma-3-27b-it:free",
            provider="openrouter",
            temperature=0.3,
        )

        # Use the same pattern as create_draft:
        # 1. Create ChatPromptTemplate from SystemMessage
        # 2. Append HumanMessagePromptTemplate with variables
        # 3. Create chain and invoke

        # Extract SystemMessage from CRITIQUE_PROMPT

        critique_system_message = SystemMessage(
            content="You are a professional editor who specializes in job applications. Provide constructive feedback."
        )

        # Create ChatPromptTemplate from SystemMessage (like line 90-94 in create_draft)
        critique_prompt_template = ChatPromptTemplate([critique_system_message])

        # Append HumanMessagePromptTemplate with variables (like line 97-124 in create_draft)
        critique_context_message = HumanMessagePromptTemplate.from_template(
            """
            # Job Description
            {job_description}

            # Current Draft
            {draft}

            Critique this draft and suggest specific improvements. Focus on:
            1. How well it targets the job requirements
            2. Professional tone and language
            3. Clarity and impact
            4. Grammar and style

            Return your critique in a constructive, actionable format.
            """,
            input_variables=["job_description", "draft"],
        )

        critique_prompt_template.append(critique_context_message)

        # Create chain (like line 129-139 in create_draft)
        critique_chain = (
            {
                "job_description": lambda x: x["job_description"],
                "draft": lambda x: x["draft"],
            }
            | critique_prompt_template
            | llm
        )

        # Invoke with validated input variables
        critique = critique_chain.invoke(
            {
                "job_description": job_description,
                "draft": draft_content,
            }
        )

        critique_content = (
            critique.content if hasattr(critique, "content") else str(critique)
        )
        logger.info("Draft critique completed")

        # Store the critique - using validated variables from top of function
        return ResultState(
            draft=state.get("draft", ""),
            feedback=state.get("feedback", ""),
            critique_feedback=str(critique_content),
            current_node="critique",
            output_data="",
            company_research_data=state.get("company_research_data", {}),
            messages=state.get("messages", []),
        )

    except Exception as e:
        logger.error(f"Error in critique_draft: {e}", exc_info=True)
        # Return state unchanged on error
        return state


def human_approval(state: ResultState) -> ResultState:
    """Human-in-the-loop checkpoint for feedback on the draft."""
    # Validate and extract all required state fields once
    draft_content = state.get("draft", "")
    critique_feedback_content = state.get("critique_feedback", "No critique available")

    # Display draft and critique for review
    print("\n" + "=" * 80)
    print("DRAFT FOR REVIEW:")
    print(draft_content)
    print("\nAUTOMATIC CRITIQUE:")
    print(critique_feedback_content)
    print("=" * 80)
    print("\nPlease provide your feedback (press Enter to continue with no changes):")

    # In a real implementation, this would be handled by the UI
    human_feedback = interrupt(
        {
            "draft": draft_content,
            "message": "Please review the draft and provide feedback (empty string to approve as-is)",
        }
    )

    print(f"Human feedback: {human_feedback}")

    return ResultState(
        draft=state.get("draft", ""),
        feedback=human_feedback,
        critique_feedback=state.get("critique_feedback", ""),
        current_node="human_approval",
        output_data="",
        company_research_data=state.get("company_research_data", {}),
        messages=state.get("messages", []),
    )


def finalize_document(state: ResultState) -> ResultState:
    """Incorporate feedback and finalize the document."""
    # Validate and extract all required state fields once
    draft_content = state.get("draft", "")
    feedback_content = state.get("feedback", "")
    critique_feedback_content = state.get("critique_feedback", "")

    if not draft_content:
        logger.warning("Missing draft in state for finalization")

    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm = llm_provider.create_langchain(
        "google/gemma-3-27b-it:free",
        provider="openrouter",
        temperature=0.3,
    )

    # Create revision chain
    revision_chain = (
        {
            "draft": lambda x: x.get("draft", ""),
            "feedback": lambda x: x.get("feedback", ""),
            "critique_feedback": lambda x: x.get("critique_feedback", ""),
        }
        | REVISION_PROMPT
        | llm
    )

    # Invoke with validated input variables
    final_content = revision_chain.invoke(
        {
            "draft": draft_content,
            "feedback": feedback_content,
            "critique_feedback": critique_feedback_content,
        }
    )

    print(
        f"Final content: {final_content.content if hasattr(final_content, 'content') else final_content}"
    )

    # Return final state using validated variables
    # Current (INCOMPLETE):

    return ResultState(
        draft=draft_content,
        feedback=feedback_content,
        critique_feedback=critique_feedback_content,
        current_node="finalize",
        output_data=(
            str(final_content.content)
            if hasattr(final_content, "content")
            else str(final_content)
        ),
        company_research_data=state.get("company_research_data", {}),
        messages=state.get("messages", []),
    )


"""
Conditional node to determine if next node should be 'draft' node or "research" node
"""


def determine_next_step(state: AppState) -> str:
    """
    Determine next workflow step based on company name presence.

    If the company name is missing within the AppState, we can't
    create the content draft and therefore redirect to the research node.

    Args:
        state: Current application state

    Returns:
        Next node name: "draft" or "research"
    """
    company_name = state.get("company_name", "")
    if not company_name:
        return "draft"
    return "research"
