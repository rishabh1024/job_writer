"""JobWorkflow orchestrator for the job application writer."""

import logging
import os
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import ConsoleCallbackHandler, LangChainTracer

from job_writing_agent.classes import DataLoadState, node_name

logger = logging.getLogger(__name__)


class JobWorkflow:
    """
    Workflow orchestrator for the job application writer.

    The workflow consists of:
    1. Data Loading: Parse resume and job description (parallel subgraph)
    2. Research: Company research and relevance filtering (subgraph)
    3. Draft Creation: Generate initial application material
    4. Critique: AI-powered feedback on the draft
    5. Human Approval: User feedback collection
    6. Finalization: Incorporate feedback and produce final output
    """

    current_time: datetime = datetime.now()
    thread_id: str = f"job_workflow_session_{current_time:%Y%m%d%H%M%S}"
    timestamp: str = current_time.strftime("%Y%m%d-%H%M%S")

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
        Get list of callbacks including LangSmith tracer with metadata.

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

        if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
            try:
                langsmith_tracer = LangChainTracer(
                    project_name=os.getenv(
                        "LANGSMITH_PROJECT", "job_application_writer"
                    )
                )
                callbacks.append(langsmith_tracer)
                logger.info("Enabled LangSmith Tracing...")
            except Exception as exc:
                logger.warning(
                    "Failed to initialize LangSmith tracer: %s. Continuing without tracing.",
                    exc,
                )
        else:
            logger.debug(
                "LangSmith tracing is not enabled (set environment variable LANGSMITH_TRACING to 'true')"
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

        return {
            "configurable": {"thread_id": self.thread_id},
            "callbacks": self._get_callbacks(),
            "run_name": f"JobAppWorkflow.{self.content}.{self.timestamp}",
            "metadata": {
                "workflow": "job_application_writer",
                "content_type": self.content,
                "session_id": self.thread_id,
            },
            "tags": ["job-application-workflow", self.content],
            "recursion_limit": 10,
        }

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
        from job_writing_agent.graph import job_app_graph

        try:
            compiled_graph = job_app_graph
        except Exception as exc:
            logger.error("Error compiling graph: %s", exc, exc_info=True)
            return None

        initial_workflow_state = self._build_initial_workflow_state()
        run_name = f"JobAppWorkflow.{self.content}.{self.timestamp}"
        config: RunnableConfig = self._build_runnable_config()

        try:
            initial_workflow_state["current_node"] = node_name.LOAD
            logger.info(
                "Starting workflow execution: %s (content_type=%s, session_id=%s)",
                run_name,
                self.content,
                self.thread_id,
            )
            graph_output = await compiled_graph.ainvoke(
                initial_workflow_state,
                config=config,
            )
            logger.info("Workflow execution completed successfully")
            return graph_output
        except Exception as exc:
            logger.error("Error running graph: %s", exc, exc_info=True)
            return None
