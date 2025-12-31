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
from typing import Tuple
from typing_extensions import Literal

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

from job_writing_agent.classes import AppState, DataLoadState
from job_writing_agent.utils.document_processing import parse_resume, get_job_description
from job_writing_agent.prompts.templates import agent_system_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper decorator to log exceptions for async methods
# ---------------------------------------------------------------------------
def log_exceptions(func):
    """Decorator to log exceptions in async functions."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            logger.error(
                "Exception in %s: %s", func.__name__, exc, exc_info=True
            )
            raise

    return wrapper


class Dataloading:
    """
    Node for loading and initializing resume and job description data.

    Methods
    -------
    set_agent_system_message(state: AppState) -> DataLoadState
        Adds the system prompt to the conversation state.
    get_resume(resume_source) -> str
        Parses a resume file and returns its plain‑text content.
    parse_job_description(job_description_source) -> Tuple[str, str]
        Parses a job description and returns its text and company name.
    load_inputs(state: DataLoadState) -> AppState
        Orchestrates loading of resume and job description.
    validate_data_load_state(state: DataLoadState)
        Ensures required fields are present in company_research_data.
    verify_inputs(state: AppState) -> Literal["load", "research"]
        Validates inputs and decides the next workflow node.
    run(state: DataLoadState) -> AppState
        Executes the loading step of the workflow.

    """
    def __init__(self):
        pass


    async def set_agent_system_message(self, state: AppState) -> DataLoadState:
        """Add the system prompt to the conversation state.

        Parameters
        ----------
        state: AppState
            Current workflow state.

        Returns
        -------
        DataLoadState
            Updated state with the system message and the next node identifier.
        """
        agent_initialization_system_message = SystemMessage(
            content=agent_system_prompt
        )
        messages = state.get("messages", [])
        messages.append(agent_initialization_system_message)
        return {
            **state,
            "messages": messages,
            "current_node": "initialize_system",
        }

    async def get_resume(self, resume_source):
        """Parse a resume file and return its plain‑text content.

        Parameters
        ----------
        resume_source: Any
            Path or file‑like object accepted by ``parse_resume``.
        """
        try:
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
                    logger.debug(
                        "Skipping empty or invalid chunk in resume: %s", chunk
                    )
            return resume_text
        except Exception as e:
            logger.error("Error parsing resume: %s", e)
            raise

    async def parse_job_description(self, job_description_source):
        """Parse a job description and return its text and company name.

        Parameters
        ----------
        job_description_source: Any
            Source accepted by ``get_job_description``.
        """
        try:
            logger.info(
                "Parsing job description from: %s", job_description_source
            )
            assert (
                job_description_source is not None
            ), "Job description source cannot be None"
            job_description_document: Document = await get_job_description(
                job_description_source
            )
            company_name = ""
            job_posting_text = ""
            if job_description_document:
                if hasattr(
                    job_description_document, "metadata"
                ) and isinstance(job_description_document.metadata, dict):
                    company_name = job_description_document.metadata.get(
                        "company_name", ""
                    )
                    if not company_name:
                        logger.warning(
                            "Company name not found in job description metadata."
                        )
                else:
                    logger.warning(
                        "Metadata attribute missing or not a dict in job "
                        "description document."
                    )
                if hasattr(job_description_document, "page_content"):
                    job_posting_text = job_description_document.page_content or ""
                    if not job_posting_text:
                        logger.info("Parsed job posting text is empty.")
                else:
                    logger.warning(
                        "page_content attribute missing in job description document."
                    )
            else:
                logger.warning(
                    "get_job_description returned None for source: %s",
                    job_description_source,
                )
            return job_posting_text, company_name
        except Exception as e:
            logger.error(
                "Error parsing job description from source '%s': %s",
                job_description_source,
                e,
                exc_info=True,
            )
            raise


    # -----------------------------------------------------------------------
    # Private helper methods used by load_inputs
    # -----------------------------------------------------------------------
    @log_exceptions
    async def _load_resume(self, resume_source) -> str:
        """Load resume content, raising if the source is missing."""
        if not resume_source:
            raise ValueError("resume_source is required")
        return await self.get_resume(resume_source)


    @log_exceptions
    async def _load_job_description(self, jd_source) -> Tuple[str, str]:
        """Load job description text and company name, raising if missing."""
        if not jd_source:
            raise ValueError("job_description_source is required")
        return await self.parse_job_description(jd_source)


    @log_exceptions
    async def _prompt_user(self, prompt_msg: str) -> str:
        """Prompt the user for input (synchronous ``input`` wrapped for async use)."""
        # In a real async UI replace ``input`` with an async call.
        return input(prompt_msg)


    async def load_inputs(self, state: DataLoadState) -> AppState:
        """Orchestrate loading of resume and job description.

        The method populates ``state['company_research_data']`` with the parsed
        resume, job description, and company name, then advances the workflow
        to the ``load_inputs`` node.
        """
        resume_src = state.get("resume_path")
        jd_src = state.get("job_description_source")

        # -------------------------------------------------------------------
        # Load job description (or prompt if missing during verification)
        # -------------------------------------------------------------------
        job_text = ""
        company_name = ""
        if jd_src:
            job_text, company_name = await self._load_job_description(jd_src)
        elif state.get("current_node") == "verify":
            job_text = await self._prompt_user(
                "Please paste the job posting in text format: "
            )

        # -------------------------------------------------------------------
        # Load resume (or prompt if missing during verification)
        # -------------------------------------------------------------------
        resume_text = ""
        if resume_src:
            resume_text = await self._load_resume(resume_src)
        elif state.get("current_node") == "verify":
            raw = await self._prompt_user(
                "Please paste the resume in text format: "
            )
            resume_text = raw

        # Populate state
        state["company_research_data"] = {
            "resume": resume_text,
            "job_description": job_text,
            "company_name": company_name,
        }
        state["current_node"] = "load_inputs"
        return state


    def validate_data_load_state(self, state: DataLoadState):
        """Ensure required fields are present in ``company_research_data``."""
        assert state.company_research_data.get(
            "resume"
        ), "Resume is missing in company_research_data"
        assert state.company_research_data.get(
            "job_description"
        ), "Job description is missing"


    def verify_inputs(self, state: AppState) -> Literal["load", "research"]:
        """Validate inputs and decide the next workflow node.

        Returns
        -------
        Literal["load", "research"]
            ``"load"`` if required data is missing, otherwise ``"research"``.
        """
        print("Verifying Inputs")
        state["current_node"] = "verify"
        logger.info("Verifying loaded inputs!")
        assert state["company_research_data"].get(
            "resume"
        ), "Resume is missing in company_research_data"
        assert state["company_research_data"].get(
            "job_description"
        ), "Job description is missing"
        if not state.get("company_research_data"):
            missing_items = []
            if not state["company_research_data"].get("resume", ""):
                missing_items.append("resume")
            if not state["company_research_data"].get("job_description", ""):
                missing_items.append("job description")
            logger.error("Missing required data: %s", ", ".join(missing_items))
            return "load"
        # Normalise values to strings
        for key in ["resume", "job_description"]:
            try:
                value = state["company_research_data"][key]
                if isinstance(value, (list, tuple)):
                    state["company_research_data"][key] = " ".join(
                        str(x) for x in value
                    )
                elif isinstance(value, dict):
                    state["company_research_data"][key] = str(value)
                else:
                    state["company_research_data"][key] = str(value)
            except Exception as e:
                logger.warning("Error converting %s to string: %s", key, e)
                raise
        return "research"


    async def run(self, state: DataLoadState) -> AppState:
        """Execute the loading step of the workflow."""
        state = await self.load_inputs(state)
        return state
