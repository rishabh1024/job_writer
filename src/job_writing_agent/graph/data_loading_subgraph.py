# -*- coding: utf-8 -*-
"""
Data Loading Subgraph Module
"""

import logging
from typing import Any, Literal

from langgraph.graph import StateGraph

from job_writing_agent.classes import (
    AgentWorkflowNodes,
    DataLoadingNodes,
    DataLoadState,
)
from job_writing_agent.classes.classes import WorkflowInput
from job_writing_agent.nodes.graph_interrupt import GraphInterrupt
from job_writing_agent.nodes.job_description_loader import JobDescriptionLoader
from job_writing_agent.nodes.resume_loader import ResumeLoader
from job_writing_agent.nodes.system_initializer import SystemInitializer
from job_writing_agent.utils.document_processing import (
    analyze_candidate_job_fit,
)

logger = logging.getLogger(__name__)

# Type alias for LangGraph partial state updates (nodes return dicts that get merged).
StateUpdate = dict[str, Any]

graph_interrupt = GraphInterrupt()


def set_agent_system_message_node(state: WorkflowInput) -> dict[str, Any]:
    """
    Initialize the agent's system message.
    #"""
    system_initializer = SystemInitializer()
    return system_initializer.set_initial_agent_state()


async def resume_loader_node(state: DataLoadState) -> DataLoadState:
    """
    Load the resume from the configured source (file path). Runs before job description parsing.

    Returns only the resume data; LangGraph merges this update with state.
    Retries on OSError/ConnectionError/TimeoutError (tenacity).

    Parameters
    ----------
    state: DataLoadState
        Current workflow state containing resume_path.

    Returns
    -------
    StateUpdate
        Partial state update with resume data in company_research_data.
    """
    resume_file_path = state.workflow_inputs.resume_file_path_
    if not resume_file_path:
        resume_text = ""
    else:
        resume_loader = ResumeLoader()
        resume_text = await resume_loader.load_resume(resume_file_path)
    logger.info(f"Resume loaded: {len(resume_text)} characters")
    company_research = state.company_research_data
    return {
        "company_research_data": company_research.model_copy(
            update={"resume": resume_text}
        )
    }


def prompt_user_for_resume_node(state: DataLoadState) -> StateUpdate:
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


def prompt_user_for_job_description_node(state: DataLoadState) -> StateUpdate:
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
    resume = (company_research.resume if company_research else "") or ""
    resume = str(resume).strip()
    if not resume:
        logger.info("Resume is empty, routing to prompt_user_for_resume")
        return "prompt_user_for_resume"
    logger.info("Resume is present, routing to load_job_description")
    return "load_job_description"


def route_after_job_description_load(
    state: DataLoadState,
) -> Literal["prompt_user_for_job_description", "candidate_job_fit_analysis"]:
    """After load_job_description: if job_description is empty, go to prompt_user_for_job_description; else go to candidate_job_fit_analysis."""
    company_research = state.company_research_data
    job_description = (
        company_research.job_description if company_research else ""
    ) or ""
    job_description = str(job_description).strip()
    if not job_description:
        logger.info(
            "Job description is empty, routing to prompt_user_for_job_description"
        )
        return "prompt_user_for_job_description"
    logger.info(
        "Job description is present, routing to candidate_job_fit_analysis"
    )
    return "candidate_job_fit_analysis"


async def job_description_loader_node(state: DataLoadState) -> StateUpdate:
    """
    Load the job description from the configured URL. Runs after resume is loaded or provided via interrupt.

    Returns job description and company name in company_research_data; LangGraph merges this update with state.
    Retries on OSError/ConnectionError/TimeoutError (tenacity).

    Parameters
    ----------
    state: DataLoadState
        Current workflow state containing job_description_url_.

    Returns
    -------
    StateUpdate
        Partial state update with job description and company name in company_research_data.
    """
    job_description_url = state.workflow_inputs.job_description_url_
    if not job_description_url:
        job_description_text = ""
        company_name = ""
    else:
        job_description_loader = JobDescriptionLoader()
        (
            job_description_text,
            company_name,
        ) = await job_description_loader.parse_job_description(
            job_description_url
        )
    company_research = state.company_research_data
    resume_text = company_research.resume or ""

    logger.info(
        f"Job description loaded: {len(job_description_text)} characters, "
        f"company: {company_name}"
    )
    return {
        "company_research_data": company_research.model_copy(
            update={
                "resume": resume_text,
                "job_description": job_description_text,
                "company_name": company_name,
            }
        )
    }


async def candidate_job_fit_analysis_node(
    state: DataLoadState,
) -> StateUpdate:
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
    StateUpdate
        Partial state update with candidate_job_fit_analysis in company_research_data
        and next_node set to NodeName.RESEARCH for main graph routing.
    """
    company_research = state.company_research_data
    resume_text = company_research.resume or ""
    job_description = company_research.job_description or ""
    company_name = company_research.company_name or ""

    # Validate inputs (should always pass due to routing, but log if not)
    if not resume_text.strip():
        logger.warning("Resume is empty in candidate_job_fit_analysis_node")
    if not job_description.strip():
        logger.warning(
            "Job description is empty in candidate_job_fit_analysis_node"
        )

    candidate_job_fit_analysis = await analyze_candidate_job_fit(
        resume_text=resume_text,
        job_description=job_description,
        company_name=company_name,
    )

    logger.info("Candidate-job fit analysis node completed")

    return {
        "company_research_data": company_research.model_copy(
            update={"candidate_job_fit_analysis": candidate_job_fit_analysis}
        ),
        "next_node": AgentWorkflowNodes.RESEARCH,
    }


# ============================================================================
# Data Loading Subgraph
# ============================================================================

data_loading_subgraph = StateGraph(
    input_schema=WorkflowInput,
    state_schema=DataLoadState,
    output_schema=DataLoadState,
)

data_loading_subgraph.add_node(
    DataLoadingNodes.SET_AGENT_SYSTEM_MESSAGE.value,
    set_agent_system_message_node,
)

data_loading_subgraph.add_node(
    DataLoadingNodes.LOAD_RESUME.value, resume_loader_node
)

data_loading_subgraph.add_node(
    DataLoadingNodes.LOAD_JOB_DESCRIPTION.value, job_description_loader_node
)

data_loading_subgraph.add_node(
    DataLoadingNodes.PROMPT_USER_FOR_RESUME.value, prompt_user_for_resume_node
)

data_loading_subgraph.add_node(
    DataLoadingNodes.PROMPT_USER_FOR_JOB_DESCRIPTION.value,
    prompt_user_for_job_description_node,
)

data_loading_subgraph.add_node(
    DataLoadingNodes.CANDIDATE_JOB_FIT_ANALYSIS.value,
    candidate_job_fit_analysis_node,
)

data_loading_subgraph.set_entry_point(
    DataLoadingNodes.SET_AGENT_SYSTEM_MESSAGE.value
)

data_loading_subgraph.set_finish_point(
    DataLoadingNodes.CANDIDATE_JOB_FIT_ANALYSIS.value
)

data_loading_subgraph.add_edge(
    DataLoadingNodes.SET_AGENT_SYSTEM_MESSAGE.value,
    DataLoadingNodes.LOAD_RESUME.value,
)

data_loading_subgraph.add_conditional_edges(
    DataLoadingNodes.LOAD_RESUME.value,
    route_after_resume_load,
    {
        DataLoadingNodes.PROMPT_USER_FOR_RESUME.value: DataLoadingNodes.PROMPT_USER_FOR_RESUME.value,
        DataLoadingNodes.LOAD_JOB_DESCRIPTION.value: DataLoadingNodes.LOAD_JOB_DESCRIPTION.value,
    },
)

data_loading_subgraph.add_edge(
    DataLoadingNodes.PROMPT_USER_FOR_RESUME.value,
    DataLoadingNodes.LOAD_JOB_DESCRIPTION.value,
)

data_loading_subgraph.add_conditional_edges(
    DataLoadingNodes.LOAD_JOB_DESCRIPTION.value,
    route_after_job_description_load,
    {
        DataLoadingNodes.PROMPT_USER_FOR_JOB_DESCRIPTION.value: DataLoadingNodes.PROMPT_USER_FOR_JOB_DESCRIPTION.value,
        DataLoadingNodes.CANDIDATE_JOB_FIT_ANALYSIS.value: DataLoadingNodes.CANDIDATE_JOB_FIT_ANALYSIS.value,
    },
)
data_loading_subgraph.add_edge(
    DataLoadingNodes.PROMPT_USER_FOR_JOB_DESCRIPTION.value,
    DataLoadingNodes.CANDIDATE_JOB_FIT_ANALYSIS.value,
)

data_loading_workflow = data_loading_subgraph.compile(name="Data Load Subgraph")
