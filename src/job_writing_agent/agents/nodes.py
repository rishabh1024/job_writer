"""
Node functions for the job application writer LangGraph.

This module contains all the node functions used in the job application
writer workflow graph, each handling a specific step in the process.
"""

import logging
from datetime import datetime

from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage

from ..classes.classes import AppState, ResearchState, ResultState, DataLoadState
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


def create_draft(state: ResearchState) -> ResultState:
    """Create initial draft of the application material."""
    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm = llm_provider.create_langchain(
        "mistralai/mistral-7b-instruct:free", provider="openrouter", temperature=0.3
    )

    # Determine which type of content we're creating
    company_background_information = state.get("company_research_data", {})

    content_category = state.get("content_category", "cover_letter")

    # Get the original resume text from state (used later if vector search is available)
    original_resume_text = company_background_information.get("resume", "")

    try:
        # Not yet implemented
        if state.get("vector_store"):
            vector_store = state.get("vector_store")

            # Extract key requirements from job description
            prompt = PERSONA_DEVELOPMENT_PROMPT | llm | StrOutputParser()

            if company_background_information:
                key_requirements = prompt.invoke(
                    {
                        "job_description": company_background_information[
                            "job_description"
                        ]
                    }
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
            # Combine highly relevant parts with full resume text
            resume_text = f"""
            # Most Relevant Experience
            {highly_relevant_resume}

            # Full Resume
            {original_resume_text}
            """
            # Update the company_background_information with the enhanced resume
            company_background_information["resume"] = resume_text
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
            Below is the Job Description, Candidate Resume, and Company Research Data enclosed in triple backticks.

            **Job Description:**

            '''
            {current_job_role}
            '''
            
            **Candidate Resume:**

            '''
            {candidate_resume}
            '''
            
            **Company Research Data:**
            
            '''
            {company_research_data}
            '''
            """,
        input_variables=[
            "current_job_role",
            "company_research_data",
            "candidate_resume",
        ],
    )

    FirstDraftGenerationPromptTemplate.append(CurrentSessionContextMessage)

    # Invoke the chain with the appropriate inputs
    draft_generation_chain = (
        (
            {
                "current_job_role": lambda x: x["current_job_role"],
                "company_research_data": lambda x: x["company_research_data"],
                "candidate_resume": lambda x: x["candidate_resume"],
            }
        )
        | FirstDraftGenerationPromptTemplate
        | llm
    )

    # Prepare the inputs
    application_background_data = {
        "current_job_role": company_background_information["job_description"],
        "company_research_data": company_background_information[
            "company_research_data_summary"
        ],
        "candidate_resume": company_background_information["resume"],
    }

    response = draft_generation_chain.invoke(application_background_data)
    logger.info(f"Draft has been created: {response.content}")
    app_state = ResultState(
        draft=response.content,
        feedback="",
        critique_feedback="",
        current_node="create_draft",
        company_research_data=company_background_information,
        output_data={},
    )

    return app_state


def critique_draft(state: ResultState) -> ResultState:
    """
    Critique the draft for improvements.
    Provides external evaluation focusing on job requirements, tone, clarity, and style.
    """
    try:
        logger.info("Critiquing draft...")

        # Create LLM inside function (lazy initialization)
        llm_provider = LLMFactory()
        llm = llm_provider.create_langchain(
            "mistralai/mistral-7b-instruct:free", provider="openrouter", temperature=0.3
        )

        job_description = str(state["company_research_data"].get("job_description", ""))
        draft = str(state.get("draft", ""))

        # Debug logging to verify values
        logger.debug(f"Job description length: {len(job_description)}")
        logger.debug(f"Draft length: {len(draft)}")

        if not job_description or not draft:
            logger.warning("Missing job_description or draft in state")
            # Return state with empty feedback
            return ResultState(
                draft=draft,
                feedback="",
                critique_feedback="",
                current_node="critique",
                company_research_data=state["company_research_data"],
                output_data=state["output_data"],
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
        CritiquePromptTemplate = ChatPromptTemplate([critique_system_message])

        # Append HumanMessagePromptTemplate with variables (like line 97-124 in create_draft)
        CritiqueContextMessage = HumanMessagePromptTemplate.from_template(
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

        CritiquePromptTemplate.append(CritiqueContextMessage)

        # Create chain (like line 129-139 in create_draft)
        critique_chain = (
            {
                "job_description": lambda x: x["job_description"],
                "draft": lambda x: x["draft"],
            }
            | CritiquePromptTemplate
            | llm
        )

        # Invoke with input variables (like line 150 in create_draft)
        critique = critique_chain.invoke(
            {
                "job_description": job_description,
                "draft": draft,
            }
        )

        critique_content = (
            critique.content if hasattr(critique, "content") else str(critique)
        )
        logger.info("Draft critique completed")

        # Store the critique for reference during revision
        app_state = ResultState(
            draft=state["draft"],
            feedback=state["feedback"],
            critique_feedback=critique_content,
            current_node="critique",
            company_research_data=state["company_research_data"],
            output_data=state["output_data"],
        )
        return app_state

    except Exception as e:
        logger.error(f"Error in critique_draft: {e}", exc_info=True)
        # Return state unchanged on error
        return state


def human_approval(state: ResultState) -> ResultState:
    """Human-in-the-loop checkpoint for feedback on the draft."""
    # This is a placeholder function that would be replaced by actual UI interaction
    print("\n" + "=" * 80)
    print("DRAFT FOR REVIEW:")
    print(state["draft"])
    print("\nAUTOMATIC CRITIQUE:")
    print(state.get("critique_feedback", "No critique available"))
    print("=" * 80)
    print("\nPlease provide your feedback (press Enter to continue with no changes):")

    # In a real implementation, this would be handled by the UI
    human_feedback = input()
    result_state = ResultState(
        draft=state["draft"],
        feedback=human_feedback,
        critique_feedback=state["critique_feedback"],
        current_node="human_approval",
        company_research_data=state["company_research_data"],
        output_data=state["output_data"],
    )
    return result_state


def finalize_document(state: ResultState) -> DataLoadState:
    """Incorporate feedback and finalize the document."""

    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm = llm_provider.create_langchain(
        "mistralai/mistral-7b-instruct:free", provider="openrouter", temperature=0.3
    )

    # Create chain like in critique_draft (line 229-236)
    revision_chain = (
        {
            "draft": lambda x: x["draft"],
            "feedback": lambda x: x["feedback"],
            "critique_feedback": lambda x: x["critique_feedback"],
        }
        | REVISION_PROMPT
        | llm
    )

    print(f"revision_chain: {revision_chain}")

    # Invoke with input variables (like line 239 in critique_draft)
    final_content = revision_chain.invoke(
        {
            "draft": state["draft"],
            "feedback": state["feedback"],
            "critique_feedback": state["critique_feedback"],
        }
    )

    app_state = DataLoadState(
        draft=state["draft"],
        feedback=state["feedback"],
        critique_feedback=state["critique_feedback"],
        company_research_data=state["company_research_data"],
        current_node="finalize",
        output_data=final_content.content
        if hasattr(final_content, "content")
        else str(final_content),
    )
    return app_state


"""
Conditional node to determine if next node should be 'draft' node or "research" node
"""


def determine_next_step(state: AppState) -> str:
    """If the company name is missing within the AppState, we can't
    create the content draft and therefore redirected to the research node."""
    if not state["company_name"]:
        return "draft"
    return "research"
