"""
Graph module for LangGraph workflow definitions.

This module contains the compiled graphs for the job application workflow,
exported for use by LangGraph API and internal orchestration.
"""

from job_writing_agent.graph.agent_workflow_graph import (
    job_app_graph,
)

__all__ = ["job_app_graph"]
