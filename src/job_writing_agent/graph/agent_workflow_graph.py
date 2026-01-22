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
from langgraph.graph.state import CompiledStateGraph

from job_writing_agent.agents.nodes import (
    create_draft,
    critique_draft,
    finalize_document,
    human_approval,
)
from job_writing_agent.classes import (
    DataLoadState,
    NodeName,
    dataload_to_research_adapter,
)
from job_writing_agent.nodes.data_loading_workflow import data_loading_workflow
from job_writing_agent.nodes.research_workflow import research_workflow

logger = logging.getLogger(__name__)


def _route_after_load(state: DataLoadState) -> str:
    """
    Route based on next_node set by data loading subgraph.

    The data loading subgraph sets next_node to either NodeName.LOAD
    (if validation fails) or NodeName.RESEARCH (if validation passes).

    Parameters
    ----------
    state : DataLoadState
        Current workflow state.

    Returns
    -------
    str
        Next node name: NodeName.LOAD or NodeName.RESEARCH.
    """
    next_node = state.get("next_node", NodeName.RESEARCH)
    logger.info(f"Routing after load: {next_node}")
    return next_node


def build_job_app_graph() -> CompiledStateGraph:
    """
    Build and compile the job application workflow graph.

    This function creates the graph structure independent of runtime inputs.
    Actual runtime values (resume, job description) come from the state
    passed during invocation.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph state machine ready for execution.
    """
    graph = StateGraph(DataLoadState)

    # Add nodes
    graph.add_node(NodeName.LOAD, data_loading_workflow)
    graph.add_node(NodeName.RESEARCH_SUBGRAPH_ADAPTER, dataload_to_research_adapter)
    graph.add_node(NodeName.RESEARCH, research_workflow)
    graph.add_node(NodeName.CREATE_DRAFT, create_draft)
    graph.add_node(NodeName.CRITIQUE, critique_draft)
    graph.add_node(NodeName.HUMAN_APPROVAL, human_approval)
    graph.add_node(NodeName.FINALIZE, finalize_document)

    # Set entry and exit
    graph.set_entry_point(NodeName.LOAD)
    graph.set_finish_point(NodeName.FINALIZE)

    # Add conditional edge for routing after data loading
    graph.add_conditional_edges(
        NodeName.LOAD,
        _route_after_load,
        {
            NodeName.LOAD: NodeName.LOAD,
            NodeName.RESEARCH: NodeName.RESEARCH_SUBGRAPH_ADAPTER,
        },
    )

    # Add sequential edges for main workflow
    graph.add_edge(NodeName.RESEARCH_SUBGRAPH_ADAPTER, NodeName.RESEARCH)
    graph.add_edge(NodeName.RESEARCH, NodeName.CREATE_DRAFT)
    graph.add_edge(NodeName.CREATE_DRAFT, NodeName.CRITIQUE)
    graph.add_edge(NodeName.CRITIQUE, NodeName.HUMAN_APPROVAL)
    graph.add_edge(NodeName.HUMAN_APPROVAL, NodeName.FINALIZE)

    return graph.compile()


# Export at module level for LangGraph API deployment
job_app_graph = build_job_app_graph()
