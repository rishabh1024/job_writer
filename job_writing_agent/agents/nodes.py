"""
Node functions for the job application writer LangGraph.

This module contains all the node functions used in the job application
writer workflow graph, each handling a specific step in the process.
"""

import logging
from datetime import datetime

from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..classes.classes import AppState, ResearchState
from ..prompts.templates import (
    CRITIQUE_PROMPT,
    PERSONA_DEVELOPMENT_PROMPT,
    COVER_LETTER_PROMPT,
    REVISION_PROMPT,
    BULLET_POINTS_PROMPT,
    LINKEDIN_NOTE_PROMPT,
)
from ..utils.llm_provider_factory import LLMFactory

logger = logging.getLogger(__name__)
# Constants
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")

llm_provider = LLMFactory()

llm = llm_provider.create_langchain(
    "qwen/qwen3-4b:free", provider="openrouter", temperature=0.3
)


def create_draft(state: ResearchState) -> AppState:
    """Create initial draft of the application material."""
    # Determine which type of content we're creating
    current_application_session = state.get("company_research_data", {})

    content_category = state.get("content_category", "cover_letter")

    try:
        if state.get("vector_store"):
            vector_store = state.get("vector_store")

            # Extract key requirements from job description
            prompt = PERSONA_DEVELOPMENT_PROMPT | llm | StrOutputParser()

            if current_application_session:
                key_requirements = prompt.invoke(
                    {"job_description": current_application_session["job_description"]}
                )
            else:
                return key_requirements

            if not key_requirements:
                print("Warning: No key requirements found in the job description.")
                return state

            # Use the key requirements to query for the most relevant resume parts
            namespace = f"resume_{state['session_id']}"
            relevant_docs = vector_store.retrieve_similar(
                query=key_requirements, namespace=namespace, k=3
            )

            # Use these relevant sections with higher weight in the draft creation
            highly_relevant_resume = "\n".join(
                [doc.page_content for doc in relevant_docs]
            )
            resume_text = f"""
            # Most Relevant Experience
            {highly_relevant_resume}

            # Full Resume
            {resume_text}
            """
    except Exception as e:
        logger.warning(f"Could not use vector search for relevant resume parts: {e}")
        # Continue with regular resume text

    # Select the appropriate prompt template based on application type and persona
    logger.info(f"The candidate wants the Agent to assist with : {content_category}")
    if content_category == "bullets":
        FirstDraftGenerationPromptTemplate = ChatPromptTemplate([BULLET_POINTS_PROMPT])
    elif content_category == "linkedin_connect_request":
        FirstDraftGenerationPromptTemplate = ChatPromptTemplate([LINKEDIN_NOTE_PROMPT])
    else:
        FirstDraftGenerationPromptTemplate = ChatPromptTemplate([COVER_LETTER_PROMPT])

    # Create the draft using the selected prompt template
    CurrentSessionContextMessage = HumanMessagePromptTemplate.from_template(
        """
            Below is the Job Description and Resume enclosed in triple backticks.

            Job Description and Resume:

            ```
            {current_job_role}

            ```
            Use the Company Research Data below in to create a cover letter that highlights the match between my qualifications and the job requirements and aligns with the company's values and culture.
            Company Research Data:
            #company_research_data

            Create a cover letter that highlights the match between my qualifications and the job requirements.
            """,
        input_variables=["current_job_role", "company_research_data"],
    )

    FirstDraftGenerationPromptTemplate.append(CurrentSessionContextMessage)

    # Invoke the chain with the appropriate inputs
    chain = (
        (
            {
                "current_job_role": lambda x: x["current_job_role"],
                "company_research_data": lambda x: x["company_research_data"],
            }
        )
        | FirstDraftGenerationPromptTemplate
        | llm
    )

    # Prepare the inputs
    inputs = {
        "current_job_role": current_application_session["job_description"],
        "company_research_data": current_application_session["tavily_search"],
    }

    response = chain.invoke(inputs)
    logger.info(f"Draft has been created: {response}")
    state["draft"] = response
    return state


def critique_draft(state: AppState) -> AppState:
    """Critique the draft for improvements."""
    critique = llm.invoke(
        CRITIQUE_PROMPT.format(
            job_description=state["job_description"][0], draft=state["draft"]
        )
    )

    # Store the critique for reference during human feedback
    state["critique"] = critique
    return state


def human_approval(state: AppState) -> AppState:
    """Human-in-the-loop checkpoint for feedback on the draft."""
    # This is a placeholder function that would be replaced by actual UI interaction
    print("\n" + "=" * 80)
    print("DRAFT FOR REVIEW:")
    print(state["draft"])
    print("\nAUTOMATIC CRITIQUE:")
    print(state.get("critique", "No critique available"))
    print("=" * 80)
    print("\nPlease provide your feedback (press Enter to continue with no changes):")

    # In a real implementation, this would be handled by the UI
    feedback = input()
    state["feedback"] = feedback
    return state


def finalize_document(state: AppState) -> AppState:
    """Incorporate feedback and finalize the document."""
    if not state["feedback"].strip():
        state["final"] = state["draft"]
        return state

    final = llm.invoke(
        REVISION_PROMPT.format(draft=state["draft"], feedback=state["feedback"])
    )

    state["final"] = final
    return state


"""
Conditional node to determine if next node should be 'draft' node or "research" node
"""


def determine_next_step(state: AppState) -> str:
    """If the company name is missing within the AppState, we can't
    create the content draft and therefore redirected to the research node."""
    if not state["company_name"]:
        return "draft"
    return "research"
