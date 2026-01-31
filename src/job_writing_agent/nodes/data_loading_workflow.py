# -*- coding: utf-8 -*-
"""
Data Loading Workflow Module

This module defines the data loading subgraph workflow, including all node
functions and the subgraph definition. It uses the separate loader classes
(ResumeLoader, JobDescriptionLoader, SystemInitializer) following the
Single Responsibility Principle.
"""

import logging
from typing import Any, Literal

from langgraph.graph import StateGraph

from job_writing_agent.classes import DataLoadState, CompanyResearchData
from job_writing_agent.nodes.graph_interrupt import GraphInterrupt
from job_writing_agent.nodes.resume_loader import ResumeLoader
from job_writing_agent.nodes.job_description_loader import JobDescriptionLoader
from job_writing_agent.nodes.system_initializer import SystemInitializer
from job_writing_agent.utils.document_processing import analyze_candidate_job_fit
from job_writing_agent.utils.logging.logging_decorators import log_async

logger = logging.getLogger(__name__)

graph_interrupt = GraphInterrupt()


# ============================================================================
# Data Loading Subgraph Node Functions
# ============================================================================


@log_async
async def set_agent_system_message_node(state: DataLoadState) -> dict[str, Any]:
    """
    Node function to initialize system message in workflow state.

    This node wraps the SystemInitializer.set_agent_system_message method
    for use in the LangGraph workflow.

    Parameters
    ----------
    state: DataLoadState
        Current workflow state.

    Returns
    -------
    DataLoadState
        Updated state with system message added to messages list.
    """
    initializer = SystemInitializer()
    return await initializer.set_agent_system_message(state)


@log_async
async def load_resume_node(state: DataLoadState) -> dict[str, Any]:
    """
    Load the resume from the configured source (file path). Runs before job description parsing.

    Returns only the resume data; LangGraph merges this update with state.

    Parameters
    ----------
    state: DataLoadState
        Current workflow state containing resume_path.

    Returns
    -------
    dict[str, Any]
        Partial state update with resume data in company_research_data.
    """
    resume_src = state.workflow_inputs.resume_file_path_
    if not resume_src:
        resume_text = ""
    else:
        loader = ResumeLoader()
        resume_text = await loader.load_resume(resume_src)
    logger.info(f"Resume loaded: {len(resume_text)} characters")
    cr = state.company_research_data or CompanyResearchData()
    return {"company_research_data": cr.model_copy(update={"resume": resume_text})}


def prompt_user_for_resume_node(state: DataLoadState) -> dict[str, Any]:
    """
    Prompt user to provide resume manually via chat (paste text).

    Used when resume extraction failed or no path was given. The workflow
    pauses; the frontend shows the interrupt payload so the user can paste
    their resume. The client resumes with Command(resume=user_input). If the
    user sends empty input, we return {} so the router sends execution back here.
    """
    logger.info("Resume missing or empty, prompting user to paste resume via chat")
    return graph_interrupt.request_input_for_field(
        state, "resume", "Please paste your resume in text format:", "resume"
    )


def prompt_user_for_job_description_node(state: DataLoadState) -> dict[str, Any]:
    """
    Prompt user to provide job description manually via chat (paste text).

    Used when job description extraction failed or no URL was given. The workflow
    pauses; the frontend shows the interrupt so the user can paste the job
    description. The client resumes with Command(job_description=user_input).
    If the user sends empty input, we return {} so the router sends execution back here.
    """
    logger.info("Job description missing or empty, prompting user to paste via chat")
    return graph_interrupt.request_input_for_field(
        state, "job_description", "Please paste the job description:", "job description"
    )


def route_after_resume_load(
    state: DataLoadState,
) -> Literal["prompt_user_for_resume", "load_job_description"]:
    """After load_resume: if resume is empty, go to prompt_user_for_resume; else go to load_job_description."""
    cr = state.company_research_data
    resume = (cr.resume if cr else "") or ""
    resume = str(resume).strip()
    if not resume:
        logger.info("Resume is empty, routing to prompt_user_for_resume")
        return "prompt_user_for_resume"
    logger.info("Resume is present, routing to load_job_description")
    return "load_job_description"


def route_after_job_load(
    state: DataLoadState,
) -> Literal["prompt_user_for_job_description", "candidate_job_fit_analysis"]:
    """After load_job_description: if job_description is empty, go to prompt_user_for_job_description; else go to candidate_job_fit_analysis."""
    cr = state.company_research_data
    job_desc = (cr.job_description if cr else "") or ""
    job_desc = str(job_desc).strip()
    if not job_desc:
        logger.info(
            "Job description is empty, routing to prompt_user_for_job_description"
        )
        return "prompt_user_for_job_description"
    logger.info("Job description is present, routing to candidate_job_fit_analysis")
    return "candidate_job_fit_analysis"


@log_async
async def load_job_description_node(state: DataLoadState) -> dict[str, Any]:
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
    jd_src = state.workflow_inputs.job_description_url_
    if not jd_src:
        job_text = ""
        company_name = ""
    else:
        loader = JobDescriptionLoader()
        job_text, company_name = await loader.load_job_description(jd_src)
    resume_text = (
        state.company_research_data.resume if state.company_research_data else ""
    )

    logger.info(
        f"Job description loaded: {len(job_text)} characters, company: {company_name}"
    )
    cr = state.company_research_data or CompanyResearchData()
    return {
        "company_research_data": cr.model_copy(
            update={
                "resume": resume_text,
                "job_description": job_text,
                "company_name": company_name,
            }
        )
    }


@log_async
async def candidate_job_fit_analysis_node(state: DataLoadState) -> dict[str, Any]:
    """
    Analyze candidate-job fit using DSPy after resume and job description are loaded.

    Uses the resume and job description to generate actionable insights
    for downstream content generation (cover letter, bullets, LinkedIn note).

    Parameters
    ----------
    state: DataLoadState
        Current workflow state with resume and job description loaded.

    Returns
    -------
    dict[str, Any]
        Partial state update with candidate_job_fit_analysis in company_research_data
        and next_node set to "research".
    """
    cr = state.company_research_data or CompanyResearchData()
    resume_text = cr.resume or ""
    job_description = cr.job_description or ""
    company_name = cr.company_name or ""

    # Validate inputs (should always pass due to routing, but log if not)
    if not resume_text.strip():
        logger.warning("Resume is empty in candidate_job_fit_analysis_node")
    if not job_description.strip():
        logger.warning("Job description is empty in candidate_job_fit_analysis_node")

    # Perform analysis
    analysis = await analyze_candidate_job_fit(
        resume_text=resume_text,
        job_description=job_description,
        company_name=company_name,
    )

    logger.info("Candidate-job fit analysis node completed")

    return {
        "company_research_data": cr.model_copy(
            update={"candidate_job_fit_analysis": analysis}
        ),
        "next_node": "research",
    }


# ============================================================================
# Data Loading Subgraph Definition
# ============================================================================

# Create data loading subgraph
data_loading_subgraph = StateGraph(DataLoadState)

# Add subgraph nodes
data_loading_subgraph.add_node(
    "set_agent_system_message", set_agent_system_message_node
)
data_loading_subgraph.add_node("load_resume", load_resume_node)
data_loading_subgraph.add_node("load_job_description", load_job_description_node)
data_loading_subgraph.add_node("prompt_user_for_resume", prompt_user_for_resume_node)
data_loading_subgraph.add_node(
    "prompt_user_for_job_description", prompt_user_for_job_description_node
)
data_loading_subgraph.add_node(
    "candidate_job_fit_analysis", candidate_job_fit_analysis_node
)

# Add subgraph edges
data_loading_subgraph.set_entry_point("set_agent_system_message")
data_loading_subgraph.set_finish_point("candidate_job_fit_analysis")
data_loading_subgraph.add_edge("set_agent_system_message", "load_resume")
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
