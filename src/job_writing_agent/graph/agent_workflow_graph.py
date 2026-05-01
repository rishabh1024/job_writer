"""
Job Application Workflow Graph Definition.

This module defines the LangGraph state machine for the job application
writing workflow. The graph is exported at module level for LangGraph API
deployment.

Workflow Structure:
    load → to_research_adapter → research → create_draft → critique → human_approval → finalize
"""

import logging

from langgraph.graph import StateGraph

from job_writing_agent.agents.nodes import (
    create_draft,
    critique_draft,
    finalize_document,
    human_approval,
)
from job_writing_agent.classes import (
    DataLoadState,
    dataload_to_research_adapter,
    node_name,
)
from job_writing_agent.nodes.data_loading_workflow import data_loading_workflow
from job_writing_agent.nodes.research_workflow import research_workflow

logger = logging.getLogger(__name__)


def _route_after_load(state: DataLoadState) -> str:
    """
    Route based on next_node set by data loading subgraph.

    The data loading subgraph sets next_node to either node_name.LOAD
    (if validation fails) or NodeName.RESEARCH (if validation passes).

    Parameters
    ----------
    state : DataLoadState
        Current workflow state.

    Returns
    -------
    str
        Next node name: node_name.LOAD or node_name.RESEARCH.
    """
    next_node = state.get("next_node", node_name.RESEARCH)
    logger.info(f"Routing after load: {next_node}")
    return next_node


"""
Build and compile the job application workflow graph.

This function creates the graph structure independent of runtime inputs.
Actual runtime values (resume, job description) come from the state
passed during invocation.

"""

graph = StateGraph(DataLoadState)

# Add nodes
graph.add_node(node_name.LOAD, data_loading_workflow)
graph.add_node(
    node_name.RESEARCH_SUBGRAPH_ADAPTER, dataload_to_research_adapter
)
graph.add_node(node_name.RESEARCH, research_workflow)
graph.add_node(node_name.CREATE_DRAFT, create_draft)
graph.add_node(node_name.CRITIQUE, critique_draft)
graph.add_node(node_name.HUMAN_APPROVAL, human_approval)
graph.add_node(node_name.FINALIZE, finalize_document)

# Set entry and exit
graph.set_entry_point(node_name.LOAD)
graph.set_finish_point(node_name.FINALIZE)

# Add conditional edge for routing after data loading
graph.add_conditional_edges(
    node_name.LOAD,
    _route_after_load,
    {
        node_name.LOAD: node_name.LOAD,
        node_name.RESEARCH: node_name.RESEARCH_SUBGRAPH_ADAPTER,
    },
)

# Add sequential edges for main workflow
graph.add_edge(node_name.RESEARCH_SUBGRAPH_ADAPTER, node_name.RESEARCH)
graph.add_edge(node_name.RESEARCH, node_name.CREATE_DRAFT)
graph.add_edge(node_name.CREATE_DRAFT, node_name.CRITIQUE)
graph.add_edge(node_name.CRITIQUE, node_name.HUMAN_APPROVAL)
graph.add_edge(node_name.HUMAN_APPROVAL, node_name.FINALIZE)

# Export at module level for LangGraph API deployment
job_app_graph = graph.compile(name="Job Application Workflow")
