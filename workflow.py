"""
Workflow runner for the job application writer.

This module provides functions for running the job application 
writer graph in both interactive and batch modes.
"""

import asyncio
import argparse
import sys

from datetime import datetime
from langchain_core.tracers import ConsoleCallbackHandler
from langgraph.graph import StateGraph
from langfuse import Langfuse


from job_writer.nodes import Dataloading
from job_writer.nodes.research_workflow import research_workflow
from job_writer.classes import AppState, DataLoadState
from job_writer.agents.nodes import (
    create_draft,
    critique_draft,
    finalize_document,
    human_approval,
)
from job_writer.nodes import (
    generate_variations,
    self_consistency_vote
)
 

class JobWorkflow:
    """
    Workflow runner for the job application writer.
    Args:
        resume: Resume text or file path
        job_description: Job description text or URL
        content:
        Type of application material to generate
        model_config: Configuration for language models
    """
   
#   
    def __init__(self, resume=None, job_description_source=None, content=None, model_configuration=None):
        """Initialize the Writing Workflow."""
        print(f"Initializing Workflow for {content}")
        self.resume = resume
        self.job_description_source = job_description_source
        self.content = content
        self.model_configuration = model_configuration

        # Initialize the app state
        self.app_state = AppState(
            resume_path=resume,
            job_description_source=job_description_source,
            company_research_data=None,
            draft="",
            feedback="",
            final="",
            content=content,
            current_node=""
        )

        self.__init__nodes()
        self._build_workflow()
        
        self.langfuse = Langfuse()


    def __init__nodes(self):
        self.dataloading = Dataloading()
        # self.createdraft = create_draft()


    def _build_workflow(self):
        # Build the graph with config
        self.job_app_graph = StateGraph(DataLoadState)


        self.job_app_graph.add_node("initialize_system", self.dataloading.system_setup)
        self.job_app_graph.add_node("load", self.dataloading.run)
        # self.job_app_graph.add_node("build_persona", select_persona)


        # Add research workflow as a node
        self.job_app_graph.add_node("research", research_workflow)
        self.job_app_graph.add_node("create_draft", create_draft)
        self.job_app_graph.add_node("variations", generate_variations)
        self.job_app_graph.add_node("self_consistency", self_consistency_vote)
        self.job_app_graph.add_node("critique", critique_draft)
        self.job_app_graph.add_node("human_approval", human_approval)
        self.job_app_graph.add_node("finalize", finalize_document)

        self.job_app_graph.set_entry_point("initialize_system")
        self.job_app_graph.set_finish_point("finalize")

        self.job_app_graph.add_edge("initialize_system", "load")
        self.job_app_graph.add_conditional_edges("load", self.dataloading.verify_inputs)
        self.job_app_graph.add_edge("research", "create_draft")
        self.job_app_graph.add_edge("create_draft", "variations")
        self.job_app_graph.add_edge("variations", "self_consistency")
        self.job_app_graph.add_edge("self_consistency", "critique")
        self.job_app_graph.add_edge("critique", "human_approval")
        self.job_app_graph.add_edge("human_approval", "finalize")


    async def run(self) -> str | None:
        """
        Run the job application writer workflow.
        """
        # Compile the graph
        try:
            compiled_graph = self.compile()
        except Exception as e:
            print(f"Error compiling graph: {e}")
            return
         # Set up run configuration
        run_name = f"Job Application Writer - {self.app_state['content']} - {datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        config = {
            "configurable": {
                "thread_id": f"job_app_session_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "callbacks": [ConsoleCallbackHandler()],
                "run_name": run_name,
                "tags": ["job-application", self.app_state['content']]
                },
            "recursion_limit": 10
            }
        # Run the graph
        try:
            self.app_state["current_node"] = "initialize_system"
            graph_output = await compiled_graph.ainvoke(self.app_state, config=config)
        except Exception as e:
            print(f"Error running graph: {e}")
            return
        
        return graph_output
    

    def compile(self):
        """Compile the graph."""
        graph = self.job_app_graph.compile()
        return graph
    
    def print_result(self, content_type, final_content):
        """Print the final generated content to the console."""
        print("\n" + "="*80)
        print(f"FINAL {content_type.upper()}:")
        print(final_content)
        print("="*80)

    
    def save_result(self, content_type, final_content):
        """Save the final generated content to a file and return the filename."""
        output_file = f"{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"\nSaved to {output_file}")
        return output_file

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate job application materials")
    parser.add_argument("--resume", required=True, help="Path to resume file or resume text")
    parser.add_argument("--job", required=True, help="Path/URL to job description or description text")
    parser.add_argument("--type", default="cover_letter", 
                       choices=["cover_letter", "bullets", "linkedin_note"],
                       help="Type of application material to generate")
    parser.add_argument("--model", help="Ollama model to use")
    parser.add_argument("--temp", type=float, help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Configure models if specified
    model_config = {}
    if args.model:
        model_config["model_name"] = args.model
    if args.temp is not None:
        model_config["temperature"] = min(0.25, args.temp)
        model_config["precise_temperature"] = min(0.2, args.temp)


    # Initialize the workflow
    workflow = JobWorkflow(
        resume=args.resume,
        job_description_source=args.job,
        content=args.type,
        model_configuration=model_config
    )

    # Run the workflow
    result = asyncio.run(workflow.run())

    if result:
        # Print the result to the console
        workflow.print_result(args.type, result["final"])
    else:
        print("Error running workflow.")
        sys.exit(1)
    

    # Save the result to a file
    if result:
        workflow.save_result(args.type, result["final"])
    else:
        print("Error saving result.")
        sys.exit(1)

    # Print a success message
    print("Workflow completed successfully.")