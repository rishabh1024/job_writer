# -*- coding: utf-8 -*-
"""
Data Loading Workflow Module

This module defines the data loading subgraph workflow, including all node
functions and the subgraph definition. It uses the separate loader classes
(ResumeLoader, JobDescriptionLoader, SystemInitializer, ValidationHelper)
following the Single Responsibility Principle.
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, END, START

from job_writing_agent.classes import DataLoadState
from job_writing_agent.nodes.resume_loader import ResumeLoader
from job_writing_agent.nodes.job_description_loader import JobDescriptionLoader
from job_writing_agent.nodes.system_initializer import SystemInitializer
from job_writing_agent.nodes.validation_helper import ValidationHelper
from job_writing_agent.utils.logging.logging_decorators import (
    log_async,
    log_execution,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading Subgraph Node Functions
# ============================================================================


@log_async
async def set_agent_system_message_node(state: DataLoadState) -> DataLoadState:
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
async def parse_resume_node(state: DataLoadState) -> DataLoadState:
    """
    Node to parse resume in parallel with job description parsing.

    Extracts resume parsing logic for parallel execution.
    Returns only the resume data - reducer will merge with job description data.

    Parameters
    ----------
    state: DataLoadState
        Current workflow state containing resume_path.

    Returns
    -------
    DataLoadState
        Partial state update with resume data in company_research_data.
        LangGraph will merge this with other parallel updates.
    """
    loader = ResumeLoader()
    resume_src = state.get("resume_path")

    resume_text = ""
    if resume_src:
        resume_text = await loader._load_resume(resume_src)
    elif state.get("current_node") == "verify":
        resume_text = await loader._prompt_user_for_resume()

    # Return only the resume data - reducer will merge this with job description data
    logger.info(f"Resume parsed: {len(resume_text)} characters")
    return {
        "company_research_data": {"resume": resume_text},
    }


@log_async
async def parse_job_description_node(state: DataLoadState) -> DataLoadState:
    """
    Node to parse job description in parallel with resume parsing.

    Extracts job description parsing logic for parallel execution.
    Returns only the job description data - reducer will merge with resume data.

    Parameters
    ----------
    state: DataLoadState
        Current workflow state containing job_description_source.

    Returns
    -------
    DataLoadState
        Partial state update with job description and company name in
        company_research_data. LangGraph will merge this with other parallel updates.
    """
    loader = JobDescriptionLoader()
    jd_src = state.get("job_description_source")

    job_text = ""
    company_name = ""
    if jd_src:
        job_text, company_name = await loader._load_job_description(jd_src)
    elif state.get("current_node") == "verify":
        job_text = await loader._prompt_user_for_job_description()

    # Return only the job description data - reducer will merge this with resume data
    logger.info(
        f"Job description parsed: {len(job_text)} characters, company: {company_name}"
    )
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
    complete. It ensures both results are present, normalizes values to strings,
    and structures the final state.

    Normalization is performed here (not in ValidationHelper) to follow SRP:
    - This function: Aggregates and normalizes data
    - ValidationHelper: Only validates data

    Parameters
    ----------
    state: DataLoadState
        Current workflow state with parallel parsing results.

    Returns
    -------
    DataLoadState
        Updated state with normalized and structured company_research_data.
    """
    # Ensure company_research_data exists
    if "company_research_data" not in state:
        state["company_research_data"] = {}

    # Extract research data once, then get results from parallel nodes
    company_research_data = state["company_research_data"]
    resume_text = company_research_data.get("resume", "")
    job_text = company_research_data.get("job_description", "")
    company_name = company_research_data.get("company_name", "")

    # Normalize values to strings (handles list, tuple, dict, str)
    def normalize_value(value: list | tuple | dict | str | Any) -> str:
        """
        Normalize a value to a string representation.

        Args:
            value: Value to normalize (list, tuple, dict, or any other type)

        Returns:
            String representation of the value
        """
        if isinstance(value, (list, tuple)):
            return " ".join(str(x) for x in value)
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)

    # Normalize all values
    resume_text = normalize_value(resume_text) if resume_text else ""
    job_text = normalize_value(job_text) if job_text else ""
    company_name = normalize_value(company_name) if company_name else ""

    # Validate both are present (log warnings but don't fail here - validation node will handle)
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

    logger.info("Data loading results aggregated and normalized successfully")
    return state


@log_execution
def verify_inputs_node(state: DataLoadState) -> DataLoadState:
    """
    Verify that required inputs are present and set next_node for routing.

    This node wraps the ValidationHelper.verify_inputs method for use in
    the LangGraph workflow. It only validates - normalization is done in
    aggregate_data_loading_results.

    Parameters
    ----------
    state: DataLoadState
        Current workflow state with aggregated and normalized data.

    Returns
    -------
    DataLoadState
        Updated state with next_node set for routing ("load" or "research").
    """
    validator = ValidationHelper()
    return validator.verify_inputs(state)


# ============================================================================
# Data Loading Subgraph Definition
# ============================================================================

# Create data loading subgraph
data_loading_subgraph = StateGraph(DataLoadState)

# Add subgraph nodes
data_loading_subgraph.add_node(
    "set_agent_system_message", set_agent_system_message_node
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
