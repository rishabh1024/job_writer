"""Workflow runner and CLI entry point for the job application writer."""

# Standard library imports
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Any

# Third-party imports
from langchain_core.tracers import ConsoleCallbackHandler, LangChainTracer
from langchain_core.runnables import RunnableConfig

# Local imports
from job_writing_agent.classes import DataLoadState, NodeName
from job_writing_agent.graph import build_job_app_graph
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

    def __repr__(self) -> str:
        return (
            f"JobWorkflow(resume={self.resume!r}, "
            f"job_description_source={self.job_description_source!r}, "
            f"content={self.content!r})"
        )

    def _build_initial_workflow_state(self) -> DataLoadState:
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

    def _get_callbacks(self) -> list[Any]:
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
        callbacks: list[Any] = [ConsoleCallbackHandler()]

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
    
    def _build_runnable_config(self) -> RunnableConfig:
        """
        Build RunnableConfig with LangSmith tracing metadata.
        
        Creates a config with workflow-specific tags, metadata, and callbacks
        for comprehensive observability across all LLM calls.
        
        Returns
        -------
        RunnableConfig
            Configured for LangSmith tracing with content-specific metadata.
        """
        current_time = datetime.now()
        thread_id = f"job_workflow_session_{current_time:%Y%m%d%H%M%S}" 
        timestamp = current_time.strftime("%Y%m%d-%H%M%S")
        
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": self._get_callbacks(),
            "run_name": f"JobAppWorkflow.{self.content}.{timestamp}",
            "metadata": {
                "workflow": "job_application_writer",
                "content_type": self.content,
                "session_id": thread_id,
            },
            "tags": ["job-application-workflow", self.content],
            "recursion_limit": 10,
        }

    @log_execution
    @log_errors
    async def run_workflow(self) -> dict[str, Any] | None:
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
            compiled_graph = build_job_app_graph()
        except Exception as e:
            logger.error("Error compiling graph: %s", e, exc_info=True)
            return None

        # Prepare enhanced LangSmith metadata and tags
        current_time = datetime.now()
        initial_workflow_state = self._build_initial_workflow_state()
        thread_id = f"job_workflow_session_{current_time:%Y%m%d%H%M%S}"
        timestamp = current_time.strftime("%Y%m%d-%H%M%S")

        # Descriptive run name for LangSmith UI
        run_name = f"JobAppWorkflow.{self.content}.{timestamp}"

        config: RunnableConfig = self._build_runnable_config()

        try:
            initial_workflow_state["current_node"] = NodeName.LOAD
            logger.info(
                f"Starting workflow execution: {run_name} "
                f"(content_type={self.content}, session_id={thread_id})"
            )
            graph_output = await compiled_graph.ainvoke(initial_workflow_state, config=config)
            logger.info("Workflow execution completed successfully")
            return graph_output
        except Exception as e:
            logger.error("Error running graph: %s", e, exc_info=True)
            return None


def main():
    args = handle_cli()
    workflow = JobWorkflow(
        resume=args.resume,
        job_description_source=args.jd_source,
        content=args.content_type,
    )
    result = asyncio.run(workflow.run_workflow())
    if result and "output_data" in result:
        print_result(args.content_type, result.get("output_data", ""))
        save_result(args.content_type, result.get("output_data", ""))
        print("Workflow completed successfully.")
    else:
        print("Error running workflow. No output data available.")
        sys.exit(1)


if __name__ == "__main__":
    main()
