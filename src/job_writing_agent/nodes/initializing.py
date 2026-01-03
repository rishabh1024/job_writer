# -*- coding: utf-8 -*-
"""
Job Application Writer - Initialization Module

This module provides the Dataloading class responsible for loading and validating
inputs required for the job-application workflow. It handles parsing resumes and
job descriptions, managing missing inputs, and populating application state.

The module includes utilities for:
- Parsing resume files and extracting text content
- Parsing job descriptions and extracting company information
- Orchestrating input loading with validation
- Providing user prompts for missing information during verification
"""

import logging
from typing import Tuple, Optional

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, START

from job_writing_agent.classes import DataLoadState
from job_writing_agent.utils.document_processing import (
    parse_resume,
    get_job_description,
)
from job_writing_agent.prompts.templates import agent_system_prompt
from job_writing_agent.utils.logging.logging_decorators import (
    log_async,
    log_execution,
    log_errors,
)

logger = logging.getLogger(__name__)


# Note: Using centralized logging decorators from utils.logging.logging_decorators


class Dataloading:
    """
    Helper class providing utility methods for loading and parsing data.

    This class provides helper methods used by the data loading subgraph nodes.
    The actual workflow orchestration is handled by the data_loading_workflow subgraph.

    Methods
    -------
    set_agent_system_message(state: DataLoadState) -> DataLoadState
        Adds the system prompt to the conversation state.
    get_resume(resume_source) -> str
        Parses a resume file and returns its plain‑text content.
    parse_job_description(job_description_source) -> Tuple[str, str]
        Parses a job description and returns its text and company name.
    verify_inputs(state: DataLoadState) -> DataLoadState
        Validates inputs and sets next_node for routing.

    Private Methods (used by subgraph nodes)
    -----------------------------------------
    _load_resume(resume_source) -> str
        Load resume content, raising if the source is missing.
    _load_job_description(jd_source) -> Tuple[str, str]
        Load job description text and company name, raising if missing.
    _prompt_user(prompt_msg: str) -> str
        Prompt the user for input (synchronous input wrapped for async use).

    """

    def __init__(self):
        """Initialize Dataloading helper class."""
        pass

    # =======================================================================
    # System/Initialization Methods
    # =======================================================================

    @log_async
    async def set_agent_system_message(self, state: DataLoadState) -> DataLoadState:
        """Add the system prompt to the conversation state.

        Parameters
        ----------
        state: DataLoadState
            Current workflow state.

        Returns
        -------
        DataLoadState
            Updated state with the system message and the next node identifier.
        """
        agent_initialization_system_message = SystemMessage(content=agent_system_prompt)
        messages = state.get("messages", [])
        messages.append(agent_initialization_system_message)
        return {
            **state,
            "messages": messages,
            "current_node": "initialize_system",
        }

    # =======================================================================
    # Public Parsing Methods
    # =======================================================================

    @log_async
    @log_errors
    async def get_resume(self, resume_source):
        """
        Parse a resume file and return its plain‑text content.

        This method extracts text from resume chunks, handling both Document
        objects and plain strings. Empty or invalid chunks are skipped.

        Parameters
        ----------
        resume_source: Any
            Path or file‑like object accepted by ``parse_resume``.

        Returns
        -------
        str
            Plain text content of the resume.

        Raises
        ------
        AssertionError
            If resume_source is None.
        Exception
            If parsing fails.
        """
        logger.info("Parsing resume...")
        resume_text = ""
        assert resume_source is not None
        resume_chunks = parse_resume(resume_source)
        for chunk in resume_chunks:
            if hasattr(chunk, "page_content") and chunk.page_content:
                resume_text += chunk.page_content
            elif isinstance(chunk, str) and chunk:
                resume_text += chunk
            else:
                logger.debug("Skipping empty or invalid chunk in resume: %s", chunk)
        return resume_text

    @log_async
    @log_errors
    async def parse_job_description(self, job_description_source):
        """
        Parse a job description and return its text and company name.

        Extracts both the job posting text and company name from the document.
        Company name is extracted from document metadata if available.

        Parameters
        ----------
        job_description_source: Any
            Source accepted by ``get_job_description`` (URL, file path, etc.).

        Returns
        -------
        Tuple[str, str]
            A tuple of (job_posting_text, company_name).

        Raises
        ------
        AssertionError
            If job_description_source is None.
        Exception
            If parsing fails.
        """
        company_name = ""
        job_posting_text = ""

        logger.info("Parsing job description from: %s", job_description_source)
        assert job_description_source is not None, (
            "Job description source cannot be None"
        )

        job_description_document: Optional[Document] = await get_job_description(
            job_description_source
        )

        # Extract company name from metadata
        if hasattr(job_description_document, "metadata") and isinstance(
            job_description_document.metadata, dict
        ):
            company_name = job_description_document.metadata.get("company_name", "")
            if not company_name:
                logger.warning("Company name not found in job description metadata.")
        else:
            logger.warning(
                "Metadata attribute missing or not a dict in job description document."
            )

        # Extract job posting text
        if hasattr(job_description_document, "page_content"):
            job_posting_text = job_description_document.page_content or ""
            if not job_posting_text:
                logger.info("Parsed job posting text is empty.")
        else:
            logger.warning(
                "page_content attribute missing in job description document."
            )

        return job_posting_text, company_name

    @log_async
    async def get_application_form_details(self, job_description_source):
        """
        Placeholder for future method to get application form details.

        This method will be implemented to extract form fields and requirements
        from job application forms.

        Parameters
        ----------
        job_description_source: Any
            Source of the job description or application form.
        """
        # TODO: Implement form field extraction
        pass

    # =======================================================================
    # Validation Methods
    # =======================================================================

    @log_execution
    @log_errors
    def verify_inputs(self, state: DataLoadState) -> DataLoadState:
        """
        Validate inputs and set next_node for routing.

        This method validates that both resume and job description are present
        in the state, normalizes their values to strings, and sets the next_node
        field for conditional routing in the main workflow.

        Parameters
        ----------
        state: DataLoadState
            Current workflow state containing company_research_data.

        Returns
        -------
        DataLoadState
            Updated state with next_node set to "load" (if validation fails)
            or "research" (if validation passes).

        Raises
        ------
        Exception
            If normalization fails for any field.
        """
        logger.info("Verifying loaded inputs!")
        state["current_node"] = "verify"

        # Validate required fields
        company_research_data = state.get("company_research_data", {})

        if not company_research_data.get("resume"):
            logger.error("Resume is missing in company_research_data")
            state["next_node"] = "load"  # Loop back to load subgraph
            return state

        if not company_research_data.get("job_description"):
            logger.error("Job description is missing in company_research_data")
            state["next_node"] = "load"  # Loop back to load subgraph
            return state

        # Normalize values to strings
        for key in ["resume", "job_description"]:
            try:
                value = company_research_data[key]
                if isinstance(value, (list, tuple)):
                    company_research_data[key] = " ".join(str(x) for x in value)
                elif isinstance(value, dict):
                    company_research_data[key] = str(value)
                else:
                    company_research_data[key] = str(value)
            except Exception as e:
                logger.warning("Error converting %s to string: %s", key, e)
                state["next_node"] = "load"
                return state

        # All validations passed
        state["next_node"] = "research"
        logger.info("Inputs verified successfully, proceeding to research")
        return state

    # =======================================================================
    # Private Helper Methods (used by subgraph nodes)
    # =======================================================================

    @log_async
    @log_errors
    async def _load_resume(self, resume_source) -> str:
        """
        Load resume content, raising if the source is missing.

        This is a wrapper around get_resume() that validates the source first.
        Used by subgraph nodes for consistent error handling.

        Parameters
        ----------
        resume_source: Any
            Path or file-like object for the resume.

        Returns
        -------
        str
            Plain text content of the resume.

        Raises
        ------
        ValueError
            If resume_source is None or empty.
        """
        if not resume_source:
            raise ValueError("resume_source is required")
        return await self.get_resume(resume_source)

    @log_async
    @log_errors
    async def _load_job_description(self, jd_source) -> Tuple[str, str]:
        """
        Load job description text and company name, raising if missing.

        This is a wrapper around parse_job_description() that validates the source first.
        Used by subgraph nodes for consistent error handling.

        Parameters
        ----------
        jd_source: Any
            Source for the job description (URL, file path, etc.).

        Returns
        -------
        Tuple[str, str]
            A tuple of (job_posting_text, company_name).

        Raises
        ------
        ValueError
            If jd_source is None or empty.
        """
        if not jd_source:
            raise ValueError("job_description_source is required")
        return await self.parse_job_description(jd_source)

    @log_async
    @log_errors
    async def _prompt_user(self, prompt_msg: str) -> str:
        """
        Prompt the user for input (synchronous input wrapped for async use).

        This method wraps the synchronous input() function to be used in async contexts.
        In a production async UI, this would be replaced with an async input mechanism.

        Parameters
        ----------
        prompt_msg: str
            Message to display to the user.

        Returns
        -------
        str
            User input string.
        """
        # In a real async UI replace input with an async call.
        return input(prompt_msg)


# ============================================================================
# Data Loading Subgraph Nodes
# ============================================================================


@log_async
async def parse_resume_node(state: DataLoadState) -> DataLoadState:
    """
    Node to parse resume in parallel with job description parsing.

    Extracts resume parsing logic from load_inputs for parallel execution.
    Returns only the resume data - reducer will merge with job description data.
    """
    dataloading = Dataloading()
    resume_src = state.get("resume_path")

    resume_text = ""
    if resume_src:
        resume_text = await dataloading._load_resume(resume_src)
    elif state.get("current_node") == "verify":
        resume_text = await dataloading._prompt_user(
            "Please paste the resume in text format: "
        )

    # Return only the resume data - reducer will merge this with job description data
    logger.info(f"Resume parsed: {len(resume_text)} characters")
    # Return partial state update - LangGraph will merge this with other parallel updates
    return {
        "company_research_data": {"resume": resume_text},
    }


@log_async
async def parse_job_description_node(state: DataLoadState) -> DataLoadState:
    """
    Node to parse job description in parallel with resume parsing.

    Extracts job description parsing logic from load_inputs for parallel execution.
    Returns only the job description data - reducer will merge with resume data.
    """
    dataloading = Dataloading()
    jd_src = state.get("job_description_source")

    job_text = ""
    company_name = ""
    if jd_src:
        job_text, company_name = await dataloading._load_job_description(jd_src)
    elif state.get("current_node") == "verify":
        job_text = await dataloading._prompt_user(
            "Please paste the job posting in text format: "
        )

    # Return only the job description data - reducer will merge this with resume data
    logger.info(
        f"Job description parsed: {len(job_text)} characters, company: {company_name}"
    )
    # Return partial state update - LangGraph will merge this with other parallel updates
    return {
        "company_research_data": {
            "job_description": job_text,
            "company_name": company_name,
        },
    }


@log_execution
def aggregate_data_loading_results(state: DataLoadState) -> DataLoadState:
    """
    Aggregate results from parallel resume and job description parsing nodes.

    This node runs after both parse_resume_node and parse_job_description_node
    complete. It ensures both results are present and normalizes the state.
    """
    # Ensure company_research_data exists
    if "company_research_data" not in state:
        state["company_research_data"] = {}

    # Get results from parallel nodes
    resume_text = state["company_research_data"].get("resume", "")
    job_text = state["company_research_data"].get("job_description", "")
    company_name = state["company_research_data"].get("company_name", "")

    # Validate both are present
    if not resume_text:
        logger.warning("Resume text is empty after parsing")
    if not job_text:
        logger.warning("Job description text is empty after parsing")

    # Ensure final structure is correct
    state["company_research_data"] = {
        "resume": resume_text,
        "job_description": job_text,
        "company_name": company_name,
    }
    state["current_node"] = "aggregate_results"

    logger.info("Data loading results aggregated successfully")
    return state


@log_execution
def verify_inputs_node(state: DataLoadState) -> DataLoadState:
    """
    Verify that required inputs are present and set next_node for routing.

    Modified from verify_inputs to return state with next_node instead of string.
    """
    dataloading = Dataloading()
    return dataloading.verify_inputs(state)


# ============================================================================
# Data Loading Subgraph
# ============================================================================

# Create data loading subgraph
data_loading_subgraph = StateGraph(DataLoadState)

# Add subgraph nodes
dataloading_instance = Dataloading()
data_loading_subgraph.add_node(
    "set_agent_system_message", dataloading_instance.set_agent_system_message
)
data_loading_subgraph.add_node("parse_resume", parse_resume_node)
data_loading_subgraph.add_node("parse_job_description", parse_job_description_node)
data_loading_subgraph.add_node("aggregate_results", aggregate_data_loading_results)
data_loading_subgraph.add_node("verify_inputs", verify_inputs_node)

# Add subgraph edges
data_loading_subgraph.add_edge(START, "set_agent_system_message")
# Parallel execution: both nodes start after set_agent_system_message
data_loading_subgraph.add_edge("set_agent_system_message", "parse_resume")
data_loading_subgraph.add_edge("set_agent_system_message", "parse_job_description")
# Both parallel nodes feed into aggregate (LangGraph waits for both)
data_loading_subgraph.add_edge("parse_resume", "aggregate_results")
data_loading_subgraph.add_edge("parse_job_description", "aggregate_results")
# Aggregate feeds into verification
data_loading_subgraph.add_edge("aggregate_results", "verify_inputs")
# Verification ends the subgraph
data_loading_subgraph.add_edge("verify_inputs", END)

# Compile data loading subgraph
data_loading_workflow = data_loading_subgraph.compile()
