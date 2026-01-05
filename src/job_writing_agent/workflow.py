"""
Workflow runner for the job application writer.
This module provides the JobWorkflow class and CLI runner.
"""

# Standard library imports
import asyncio
import logging
import os
import sys
from datetime import datetime
from functools import cached_property
from typing import Any

# Third-party imports
from langchain_core.tracers import ConsoleCallbackHandler, LangChainTracer
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

# Local imports
from job_writing_agent.agents.nodes import (
    create_draft,
    critique_draft,
    finalize_document,
    human_approval,
)
from job_writing_agent.classes import DataLoadState, ResearchState
from job_writing_agent.nodes.data_loading_workflow import data_loading_workflow
from job_writing_agent.nodes.research_workflow import research_workflow
from job_writing_agent.utils.application_cli_interface import handle_cli
from job_writing_agent.utils.logging.logging_decorators import (
    log_errors,
    log_execution,
)
from job_writing_agent.utils.result_utils import print_result, save_result

logger = logging.getLogger(__name__)


class JobWorkflow:
    """
    Workflow orchestrator for the job application writer.

    This class coordinates the execution of the job application writing workflow,
    managing the LangGraph state machine and LangSmith tracing. It follows the
    orchestrator pattern, coordinating multiple subgraphs and nodes without
    implementing business logic itself.

    The workflow consists of:
    1. Data Loading: Parse resume and job description (parallel subgraph)
    2. Research: Company research and relevance filtering (subgraph)
    3. Draft Creation: Generate initial application material
    4. Critique: AI-powered feedback on the draft
    5. Human Approval: User feedback collection
    6. Finalization: Incorporate feedback and produce final output
    """

    def __init__(self, resume: str, job_description_source: str, content: str):
        """
        Initialize the JobWorkflow orchestrator.

        Parameters
        ----------
        resume: str
            Path to the resume file or resume text.
        job_description_source: str
            URL, file path, or text content of the job description.
        content: str
            Type of application material to generate ("cover_letter", "bullets", "linkedin_note").
        """
        self.resume = resume
        self.job_description_source = job_description_source
        self.content = content

    @cached_property
    def app_state(self) -> DataLoadState:
        """
        Get the initial application state for the workflow.

        Returns
        -------
        DataLoadState
            Initialized state dictionary with resume path, job description source,
            content type, and empty messages list.
        """

        return {
            "resume_path": self.resume,
            "job_description_source": self.job_description_source,
            "content_category": self.content,
            "current_node": "",
            "messages": [],
            "company_research_data": {},
        }

    # Conditional routing after data loading
    def route_after_load(self, state: DataLoadState) -> str:
        """
        Route based on next_node set by data loading subgraph.

        The data loading subgraph sets next_node to either "load" (if validation
        fails) or "research" (if validation passes).

        Parameters
        ----------
        state: DataLoadState
            Current workflow state.

        Returns
        -------
        str
            Next node name: "load" or "research".
        """
        next_node = state.get("next_node", "research")  # Default to research
        logger.info(f"Routing after load: {next_node}")
        return next_node

    def dataload_to_research_adapter(self, state: DataLoadState) -> ResearchState:
        """
        Adapter to convert DataLoadState to ResearchState.

        Extracts only fields needed for research workflow following the
        adapter pattern recommended by LangGraph documentation.

        Parameters
        ----------
        state: DataLoadState
            Current workflow state with loaded data.

        Returns
        -------
        ResearchState
            State formatted for research subgraph with required fields.
        """
        logger.info("Adapter for converting DataLoadState to ResearchState")

        return ResearchState(
            company_research_data=state.get("company_research_data", {}),
            attempted_search_queries=[],
            current_node="",
            content_category=state.get("content_category", ""),
            messages=state.get("messages", []),
        )

    @cached_property
    def job_app_graph(self) -> StateGraph:
        """
        Build and configure the job application workflow graph.

        This method constructs the LangGraph state machine with all nodes and edges.
        The graph is cached as a property to avoid rebuilding on each access.

        Workflow Structure:
        - Entry: Data loading subgraph (parallel resume + job description parsing)
        - Research: Company research subgraph
        - Draft Creation: Generate initial application material
        - Critique: AI feedback on draft
        - Human Approval: User feedback collection
        - Finalization: Produce final output
        - Exit: Finalize node

        Returns
        -------
        StateGraph
            Configured LangGraph state machine ready for compilation.
        """
        agent_workflow_graph = StateGraph(DataLoadState)

        # Add workflow nodes (subgraphs and individual nodes)
        agent_workflow_graph.add_node("load", data_loading_workflow)
        agent_workflow_graph.add_node(
            "to_research_adapter", self.dataload_to_research_adapter
        )
        agent_workflow_graph.add_node("research", research_workflow)
        agent_workflow_graph.add_node("create_draft", create_draft)
        agent_workflow_graph.add_node("critique", critique_draft)
        agent_workflow_graph.add_node("human_approval", human_approval)
        agent_workflow_graph.add_node("finalize", finalize_document)

        # Set entry and exit points
        agent_workflow_graph.set_entry_point("load")
        agent_workflow_graph.set_finish_point("finalize")

        agent_workflow_graph.add_conditional_edges(
            "load",
            self.route_after_load,
            {
                "load": "load",  # Loop back to load subgraph if validation fails
                "research": "to_research_adapter",  # Route to adapter first
            },
        )

        # Sequential edges for main workflow
        agent_workflow_graph.add_edge("to_research_adapter", "research")
        agent_workflow_graph.add_edge("research", "create_draft")
        agent_workflow_graph.add_edge("create_draft", "critique")
        agent_workflow_graph.add_edge("critique", "human_approval")
        agent_workflow_graph.add_edge("human_approval", "finalize")

        return agent_workflow_graph

    def _get_callbacks(self) -> list:
        """
        Get list of callbacks including LangSmith tracer with enhanced metadata.

        This method creates callback handlers for LangGraph execution, including
        LangSmith tracing with workflow-level metadata and tags for better
        observability and filtering in the LangSmith UI.

        Returns
        -------
        list
            List of callback handlers for LangGraph execution, including:
            - ConsoleCallbackHandler: Console output
            - LangChainTracer: LangSmith tracing (if enabled)
        """
        callbacks = [ConsoleCallbackHandler()]

        # Add LangSmith tracer if tracing is enabled via environment variable
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            try:
                # LangChainTracer automatically reads from environment variables:
                # - LANGSMITH_API_KEY
                # - LANGSMITH_PROJECT (optional, defaults to "default")
                # - LANGSMITH_ENDPOINT (optional, defaults to https://api.smith.langchain.com)
                langsmith_tracer = LangChainTracer(
                    project_name=os.getenv(
                        "LANGSMITH_PROJECT", "job_application_writer"
                    )
                )
                callbacks.append(langsmith_tracer)
                logger.info("LangSmith tracing enabled with metadata")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LangSmith tracer: {e}. Continuing without tracing."
                )
        else:
            logger.debug(
                "LangSmith tracing is not enabled (LANGSMITH_TRACING != 'true')"
            )

        return callbacks

    @log_execution
    @log_errors
    async def run(self) -> dict[str, Any] | None:
        """
        Execute the complete job application writer workflow.

        This method compiles the graph, configures LangSmith tracing with
        enhanced metadata, and executes the workflow. It handles errors
        gracefully and returns the final state or None if execution fails.

        Returns
        -------
        Optional[Dict[str, Any]]
            Final workflow state containing the generated application material
            in the "output_data" field, or None if execution fails.
        """
        try:
            compiled_graph = self.compile()
        except Exception as e:
            logger.error("Error compiling graph: %s", e, exc_info=True)
            return None

        # Prepare enhanced LangSmith metadata and tags
        content = self.app_state.get("content", "cover_letter")
        thread_id = f"job_app_session_{datetime.now():%Y%m%d%H%M%S}"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Enhanced metadata for better trace filtering and analysis
        metadata = {
            "workflow": "job_application_writer",
            "content_type": content,
            "session_id": thread_id,
        }

        # Enhanced tags for trace organization
        tags = [
            "job-application",
            content,
        ]

        # Descriptive run name for LangSmith UI
        run_name = f"JobAppWriter.{content}.{timestamp}"

        config = {
            "configurable": {
                "thread_id": thread_id,
                "callbacks": self._get_callbacks(),
                "run_name": run_name,
                "metadata": metadata,
                "tags": tags,
            },
            "recursion_limit": 10,
        }

        try:
            self.app_state["current_node"] = "load"
            logger.info(
                f"Starting workflow execution: {run_name} "
                f"(content_type={content}, session_id={thread_id})"
            )
            graph_output = await compiled_graph.ainvoke(self.app_state, config=config)
            logger.info("Workflow execution completed successfully")
            return graph_output
        except Exception as e:
            logger.error("Error running graph: %s", e, exc_info=True)
            return None

    @log_execution
    @log_errors
    def compile(self) -> CompiledStateGraph:
        """
        Compile the workflow graph into an executable state machine.

        Returns
        -------
        CompiledStateGraph
            Compiled LangGraph state machine ready for execution.

        Raises
        ------
        Exception
            If graph compilation fails (e.g., invalid edges, missing nodes).
        """
        compiled_graph = self.job_app_graph.compile()
        return compiled_graph


def main():
    args = handle_cli()
    workflow = JobWorkflow(
        resume=args.resume,
        job_description_source=args.job_posting,
        content=args.content_type,
    )
    result = asyncio.run(workflow.run())
    if result:
        print_result(args.content_type, result["output_data"])
        save_result(args.content_type, result["output_data"])
        print("Workflow completed successfully.")
    else:
        print("Error running workflow.")
        sys.exit(1)


if __name__ == "__main__":
    main()
