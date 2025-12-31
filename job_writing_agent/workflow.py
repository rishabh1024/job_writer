"""
Workflow runner for the job application writer.
This module provides the JobWorkflow class and CLI runner.
"""

import asyncio
import logging
import sys
from datetime import datetime
from functools import cached_property
from typing import Optional, Dict, Any

from langchain_core.tracers import ConsoleCallbackHandler
from langgraph.graph import StateGraph
from langfuse import Langfuse
from langgraph.graph.state import CompiledStateGraph

from job_writing_agent.agents.nodes import (
    create_draft,
    critique_draft,
    finalize_document,
    human_approval,
)
from job_writing_agent.classes import AppState, DataLoadState
from job_writing_agent.nodes import Dataloading, generate_variations, self_consistency_vote
from job_writing_agent.nodes.research_workflow import research_workflow
from job_writing_agent.utils.application_cli_interface import handle_cli
from job_writing_agent.utils.result_utils import print_result, save_result


logger = logging.getLogger(__name__)


class JobWorkflow:
    """
    Workflow runner for the job application writer.
    """

    def __init__(self, resume: str, job_description_source: str, content: str):
        self.resume = resume
        self.job_description_source = job_description_source
        self.content = content
        self.dataloading = Dataloading()
        self.langfuse = Langfuse()

    @cached_property
    def app_state(self) -> AppState:
        return AppState(
            resume_path=self.resume,
            job_description_source=self.job_description_source,
            company_research_data=None,
            draft="",
            feedback="",
            final="",
            content=self.content,
            current_node="",
        )

    @cached_property
    def job_app_graph(self) -> StateGraph:
        graph = StateGraph(DataLoadState)
        graph.add_node("initialize_system", self.dataloading.set_agent_system_message)
        graph.add_node("load", self.dataloading.run)
        graph.add_node("research", research_workflow)
        graph.add_node("create_draft", create_draft)
        graph.add_node("variations", generate_variations)
        graph.add_node("self_consistency", self_consistency_vote)
        graph.add_node("critique", critique_draft)
        graph.add_node("human_approval", human_approval)
        graph.add_node("finalize", finalize_document)

        graph.set_entry_point("initialize_system")
        graph.set_finish_point("finalize")
        graph.add_edge("initialize_system", "load")
        graph.add_conditional_edges("load", self.dataloading.verify_inputs)
        graph.add_edge("research", "create_draft")
        graph.add_edge("create_draft", "variations")
        graph.add_edge("variations", "self_consistency")
        graph.add_edge("self_consistency", "critique")
        graph.add_edge("critique", "human_approval")
        graph.add_edge("human_approval", "finalize")
        return graph

    async def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the job application writer workflow.
        """
        try:
            compiled_graph = self.compile()
        except Exception as e:
            logger.error("Error compiling graph: %s", e)
            return None

        run_name = f"Job Application Writer - {self.app_state['content']} - {datetime.now():%Y-%m-%d-%H%M%S}"
        config = {
            "configurable": {
                "thread_id": f"job_app_session_{datetime.now():%Y%m%d%H%M%S}",
                "callbacks": [ConsoleCallbackHandler()],
                "run_name": run_name,
                "tags": ["job-application", self.app_state["content"]],
            },
            "recursion_limit": 10,
        }
        try:
            self.app_state["current_node"] = "initialize_system"
            graph_output = await compiled_graph.ainvoke(self.app_state, config=config)
        except Exception as e:
            logger.error("Error running graph: %s", e)
            return None
        return graph_output

    def compile(self) -> CompiledStateGraph:
        """Compile the workflow graph."""
        return self.job_app_graph.compile()


def main():
    args = handle_cli()
    workflow = JobWorkflow(
        resume=args.resume,
        job_description_source=args.job_posting,
        content=args.content_type,
    )
    result = asyncio.run(workflow.run())
    if result:
        print_result(args.content_type, result["final"])
        save_result(args.content_type, result["final"])
        print("Workflow completed successfully.")
    else:
        print("Error running workflow.")
        sys.exit(1)


if __name__ == "__main__":
    main()
