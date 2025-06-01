"""
State definitions for the Job Writer LangGraph Workflow.
"""

from typing_extensions import List, Dict, Any
from langgraph.graph import MessagesState


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
    company_research_data: Dict[str, Any]
    draft: str
    feedback: str
    final: str
    content: str  # "cover_letter", "bullets", "linkedin_note"
    current_node: str


class DataLoadState(MessagesState):
    """
    State container for the job application writer workflow.
    
    Attributes:
        resume: List of text chunks from the candidate's resume
        job_description: List of text chunks from the job description
        persona: The writing persona to use ("recruiter" or "hiring_manager")
        content: Type of application material to generate
    """
    resume_path: str
    job_description_source: str
    resume: str
    job_description: str
    company_name: str
    current_node: str
    company_research_data: Dict[str, Any]


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