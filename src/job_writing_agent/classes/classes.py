"""
State definitions for the Job Writer LangGraph Workflow.
"""

from typing import Annotated
from typing_extensions import List, Dict, Any
from langgraph.graph import MessagesState
from dataclasses import dataclass


def merge_dict_reducer(
    x: Dict[str, Any] | None, y: Dict[str, Any] | None
) -> Dict[str, Any]:
    """
    Reducer function to merge two dictionaries.
    Used for company_research_data to allow parallel nodes to update it.

    Args:
        x: First dictionary (existing state or None)
        y: Second dictionary (new update or None)

    Returns:
        Merged dictionary with y taking precedence for overlapping keys
    """
    # Handle None cases - treat as empty dict
    if x is None:
        x = {}
    if y is None:
        y = {}

    # Merge dictionaries, with y taking precedence for overlapping keys
    return {**x, **y}


@dataclass
class AppState(MessagesState):
    """
    State container for the job application writer workflow.

    Attributes:
        resume: List of text chunks from the candidate's resume
        job_description: List of text chunks from the job description
        company_name: Extracted company name
        company_research_data: Additional information about the company from research
        persona: The writing persona to use ("recruiter" or "hiring_manager")
        draft: Current draft of the application material
        feedback: Human feedback on the draft
        final: Final version of the application material
        content: Type of application material to generate
    """

    resume_path: str
    job_description_source: str
    content: str  # "cover_letter", "bullets", "linkedin_note"
    current_node: str


class DataLoadState(MessagesState, total=False):
    """
    State container for the job application writer workflow.
    Includes all fields needed throughout the entire workflow.

    Attributes:
        resume: List of text chunks from the candidate's resume
        job_description: List of text chunks from the job description
        persona: The writing persona to use ("recruiter" or "hiring_manager")
        content: Type of application material to generate
        draft: Current draft of the application material
        feedback: Human feedback on the draft
        critique_feedback: Automated critique feedback
        output_data: Final output data
        next_node: Next node to route to after data loading subgraph
    """

    resume_path: str
    job_description_source: str
    content: str  # "cover_letter", "bullets", "linkedin_note"
    resume: str
    job_description: str
    company_name: str
    current_node: str
    next_node: str  # For routing after data loading subgraph
    # Use Annotated with reducer to allow parallel nodes to merge dictionary updates
    company_research_data: Annotated[Dict[str, Any], merge_dict_reducer]
    # Result fields (added for final output - optional, populated later)
    draft: str
    feedback: str
    critique_feedback: str
    output_data: str


class ResearchState(MessagesState):
    """
    State container for the job application writer workflow.
    Attributes:
        tavily_search: Dict[str, Any] Stores the results of the Tavily search
        attempted_search_queries: List of queries used extracted from the job description
        compiled_knowledge: Compiled knowledge from the research
    """

    company_research_data: Dict[str, Any]
    attempted_search_queries: List[str]
    current_node: str


class ResultState(MessagesState):
    """
    State container for the job application writer workflow.
    Attributes:
        final_result: The final generated application material
    """

    draft: str
    feedback: str
    critique_feedback: str
    current_node: str
    company_research_data: Dict[str, Any]
    output_data: str
