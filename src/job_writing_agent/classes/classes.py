"""
State definitions for the Job Writer LangGraph Workflow.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated
from typing_extensions import List, Dict, Any

from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


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


def immutable_field(existing, new):
    """Ignore updates - keep original value."""
    return existing  # Always return existing, ignore new


class WorkflowInput(BaseModel):
    """
    Input parameters for the job application writer workflow.

    Attributes:
        resume: Path to the resume file or resume text.
        job_description_source: URL, file path, or text content of the job description.
        content: Type of application material to generate ("cover_letter", "bullets", "linkedin_note").
    """

    resume_file_path_: str = Field(
        default="https://huggingface.co/datasets/Rishabh2095/"
        "resume-file-dataset/resolve/main/resume.pdf",
        description="Provide a valid path to the resume file. It can be a"
        " local file path or a url to the file).",
    )
    job_description_url_: str = Field(
        ..., description="Provide a valid link to the job description."
    )
    content_category_: str = Field(
        default="cover_letter",
        description="Choose one of the following :"
        "'cover_letter', 'bullets', or 'linkedin_note'",
    )


class CompanyResearchData(BaseModel):
    """
    Container for company research data.

    Attributes:
        company_name: Name of the company
        job_description: Text of the job description
        resume: Text of the candidate's resume
        tavily_search: Results from Tavily company research
        candidate_job_fit_analysis: DSPy analysis of resume-job alignment
    """

    company_name: str = Field(default="")
    job_description: str = Field(default="")
    resume: str = Field(default="")
    tavily_search: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_job_fit_analysis: Dict[str, Any] = Field(default_factory=dict)


class DataLoadState(BaseModel):
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

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    workflow_inputs: WorkflowInput = Field(default_factory=WorkflowInput)
    next_node: str = Field(default="")  # For routing after data loading subgraph
    current_node: str = Field(default="")
    company_research_data: CompanyResearchData = Field(
        default_factory=CompanyResearchData
    )


class ResearchState(BaseModel):
    """
    State container for the job application writer workflow.
    Attributes:
        tavily_search: Dict[str, Any] Stores the results of the Tavily search
        attempted_search_queries: List of queries used extracted from the job description
        compiled_knowledge: Compiled knowledge from the research
        content_category: Type of application material to generate
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    company_research_data: CompanyResearchData = Field(
        default_factory=CompanyResearchData
    )
    attempted_search_queries: List[str] = Field(default_factory=list)
    current_node: str = Field(default="")
    content_category: str = Field(default="cover_letter")


class ResultState(BaseModel):
    """
    State container for the job application writer workflow.
    Attributes:
        final_result: The final generated application material
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    draft: str = Field(default="")
    feedback: str = Field(default="")
    critique_feedback: str = Field(default="")
    current_node: str = Field(default="")
    company_research_data: CompanyResearchData = Field(
        default_factory=CompanyResearchData
    )
    output_data: str = Field(default="")


class NodeName(StrEnum):
    """Node names for the job application workflow graph."""

    LOAD = "load"
    RESEARCH_SUBGRAPH_ADAPTER = "to_research_adapter"
    RESEARCH = "research"
    CREATE_DRAFT = "create_draft"
    CRITIQUE = "critique"
    HUMAN_APPROVAL = "human_approval"
    FINALIZE = "finalize"


def dataload_to_research_adapter(state: DataLoadState) -> ResearchState:
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

    return ResearchState(
        company_research_data=getattr(
            state, "company_research_data", CompanyResearchData()
        )
        or CompanyResearchData(),
        attempted_search_queries=[],
        content_category=getattr(state, "content_category", ""),
        messages=getattr(state, "messages", []),
    )
