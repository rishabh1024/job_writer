# -*- coding: utf-8 -*-
"""
Data Loading Workflow Module

This module defines the data loading subgraph workflow, including all node
functions and the subgraph definition. It uses the separate loader classes
(ResumeLoader, JobDescriptionLoader, SystemInitializer) following the
Single Responsibility Principle.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy
from pydantic import HttpUrl

from job_writing_agent.classes import (
    CompanyResearchData,
    DataLoadState,
    WorkflowInput,
)
from job_writing_agent.nodes.graph_interrupt import GraphInterrupt
from job_writing_agent.utils.browser_session.src.playwright_browser import (
    AgentQLBrowser,
)
from job_writing_agent.utils.document_loader.src.pdf_document_loader import (
    PDFDocumentLoader,
)
from job_writing_agent.utils.document_loader.src.web_document_loader import (
    WebDocumentLoader,
)
from job_writing_agent.utils.document_processing import (
    analyze_candidate_job_fit,
)

logger = logging.getLogger(__name__)

graph_interrupt = GraphInterrupt()


# ============================================================================
# Data Loading Subgraph Node Functions
# ============================================================================


def agent_initial_setup_node(state: WorkflowInput) -> dict[str, Any]:
    """
    Map invoke input (``WorkflowInput``) into ``DataLoadState`` and seed messages.

    Runs on the parent graph before the data-loading subgraph. Parent uses
    ``input_schema=WorkflowInput`` and ``state_schema=DataLoadState``; flat invoke
    fields must be written to the ``workflow_inputs`` channel here.
    """

    return {
        "workflow_inputs": state,
    }


async def candidate_resume_loader_node(state: DataLoadState) -> dict[str, Any]:
    """
    Load the resume from the file path. This node runs in parallel with the job description loader node.

    Once the resume text data is extracted, it is added to the company_research_data in the state.
    """
    resume_source_path = Path(state.workflow_inputs.resume_file_path_)
    logger.debug(f"resume source path: {resume_source_path}")

    pdf_document_loader = PDFDocumentLoader()
    try:
        resume_document = pdf_document_loader.load_document(resume_source_path)
        logger.info(
            f"Resume loaded: {len(resume_document.page_content)} characters"
        )
        return {
            "company_research_data": CompanyResearchData(
                resume=resume_document.page_content
            )
        }
    except Exception as e:
        logger.exception(f"Candidate Resume Loader Node Failed. Exception: {e}")
        raise e


def prompt_user_for_resume_node(state: DataLoadState) -> dict[str, Any]:
    """
    Prompt user to provide resume manually via chat (paste text).

    Used when resume extraction failed or no path was given. The workflow
    pauses; the frontend shows the interrupt payload so the user can paste
    their resume. The client resumes with Command(resume=user_input). If the
    user sends empty input, we return {} so the router sends execution back here.
    """
    logger.info(
        "Resume missing or empty, prompting user to paste resume via chat"
    )
    return graph_interrupt.request_input_for_field(
        state, "resume", "Please paste your resume in text format:", "resume"
    )


def prompt_user_for_job_description_node(
    state: DataLoadState,
) -> dict[str, Any]:
    """
    Prompt user to provide job description manually via chat (paste text).

    Used when job description extraction failed or no URL was given. The workflow
    pauses; the frontend shows the interrupt so the user can paste the job
    description. The client resumes with Command(job_description=user_input).
    If the user sends empty input, we return {} so the router sends execution back here.
    """
    logger.info(
        "Job description missing or empty, prompting user to paste via chat"
    )
    return graph_interrupt.request_input_for_field(
        state,
        "job_description",
        "Please paste the job description:",
        "job description",
    )


def route_after_resume_load(
    state: DataLoadState,
) -> Literal["prompt_user_for_resume", "load_job_description"]:
    """After load_resume: if resume is empty, go to prompt_user_for_resume; else go to load_job_description."""
    company_research = state.company_research_data
    resume_text = (company_research.resume if company_research else "") or ""
    resume_text = str(resume_text).strip()
    if not resume_text:
        logger.info("Resume is empty, routing to prompt_user_for_resume")
        return "prompt_user_for_resume"
    logger.info("Resume is present, routing to load_job_description")
    return "load_job_description"


def route_after_job_load(
    state: DataLoadState,
) -> Literal["prompt_user_for_job_description", "candidate_job_fit_analysis"]:
    """After load_job_description: if job_description is empty, go to prompt_user_for_job_description; else go to candidate_job_fit_analysis."""
    company_research = state.company_research_data
    job_description_text = (
        company_research.job_description if company_research else ""
    ) or ""
    job_description_text = str(job_description_text).strip()
    if not job_description_text:
        logger.info(
            "Job description is empty, routing to prompt_user_for_job_description"
        )
        return "prompt_user_for_job_description"
    logger.info(
        "Job description is present, routing to candidate_job_fit_analysis"
    )
    return "candidate_job_fit_analysis"


async def job_description_loader_node(state: DataLoadState) -> dict[str, Any]:
    """
    Load the job description from the configured URL. Runs after resume is loaded or provided via interrupt.

    Returns job description and company name in company_research_data; LangGraph merges this update with state.

    Parameters
    ----------
    state: DataLoadState
        Current workflow state containing job_description_url_.

    Returns
    -------
    dict[str, Any]
        Partial state update with job description and company name in company_research_data.
    """
    job_posting_url = state.workflow_inputs.job_description_url_

    try:
        browser_session = AgentQLBrowser()
        job_description_loader = WebDocumentLoader(browser_session)
        job_description_document = await job_description_loader.load_document(
            HttpUrl(job_posting_url)
        )
        logger.info(
            f"Job description document: {type(job_description_document.page_content)}"
        )
        agent_research_context = state.company_research_data
        return {
            "company_research_data": agent_research_context.model_copy(
                update={
                    "job_description": job_description_document.page_content,
                    "company_name": job_description_document.metadata.get(
                        "company_name", ""
                    ),
                }
            )
        }
    except Exception as e:
        logger.exception(f"Job Description Loader Node Failed. Exception: {e}")
        raise e


async def candidate_job_fit_analysis_node(
    state: DataLoadState,
) -> dict[str, Any]:
    """
    Analyze candidate-job fit using DSPy after resume and job description are loaded.

    Uses the resume and job description to generate actionable insights
    for downstream content generation (cover letter, bullets, LinkedIn note).

    Parameters
    state: DataLoadState
        Current workflow state with resume and job description loaded.

    Returns
    -------
    dict[str, Any]
        Partial state update with candidate_job_fit_analysis in company_research_data
        and next_node set to "research".
    """
    company_research = state.company_research_data or CompanyResearchData()
    resume_text = company_research.resume or ""
    job_description_text = company_research.job_description or ""
    company_name = company_research.company_name or ""

    # Validate inputs (should always pass due to routing, but log if not)
    if not resume_text.strip():
        logger.warning("Resume is empty in candidate_job_fit_analysis_node")
    if not job_description_text.strip():
        logger.warning(
            "Job description is empty in candidate_job_fit_analysis_node"
        )

    # Perform analysis
    analysis = await analyze_candidate_job_fit(
        resume_text=resume_text,
        job_description=job_description_text,
        company_name=company_name,
    )

    logger.info("Candidate-job fit analysis node completed")

    return {
        "company_research_data": company_research.model_copy(
            update={"candidate_job_fit_analysis": analysis}
        ),
        "next_node": "research",
    }


# ============================================================================
# Data Loading Subgraph Definition
# ============================================================================

# Create data loading subgraph
data_loading_subgraph = StateGraph(
    state_schema=DataLoadState,
    input_schema=DataLoadState,
    output_schema=DataLoadState,
)

# Add subgraph nodes
data_loading_subgraph.add_node(
    "load_resume", candidate_resume_loader_node, retry_policy=RetryPolicy()
)
data_loading_subgraph.add_node(
    "load_job_description",
    job_description_loader_node,
    retry_policy=RetryPolicy(),
)
data_loading_subgraph.add_node(
    "prompt_user_for_resume", prompt_user_for_resume_node
)
data_loading_subgraph.add_node(
    "prompt_user_for_job_description", prompt_user_for_job_description_node
)
data_loading_subgraph.add_node(
    "candidate_job_fit_analysis", candidate_job_fit_analysis_node
)

# Add subgraph edges
data_loading_subgraph.set_entry_point("load_resume")
data_loading_subgraph.set_finish_point("candidate_job_fit_analysis")
# After resume load: prompt user to paste if empty, else continue to load job description
data_loading_subgraph.add_conditional_edges(
    "load_resume",
    route_after_resume_load,
    {
        "prompt_user_for_resume": "prompt_user_for_resume",
        "load_job_description": "load_job_description",
    },
)
data_loading_subgraph.add_edge("prompt_user_for_resume", "load_job_description")
# After job description load: prompt user to paste if empty, else analyze candidate-job fit
data_loading_subgraph.add_conditional_edges(
    "load_job_description",
    route_after_job_load,
    {
        "prompt_user_for_job_description": "prompt_user_for_job_description",
        "candidate_job_fit_analysis": "candidate_job_fit_analysis",
    },
)
data_loading_subgraph.add_edge(
    "prompt_user_for_job_description", "candidate_job_fit_analysis"
)

# Compile data loading subgraph
data_loading_workflow = data_loading_subgraph.compile(name="Data Load Subgraph")
