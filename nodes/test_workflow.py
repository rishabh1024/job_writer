from typing_extensions import List, Dict, Any, Optional
from langgraph.graph import MessagesState, StateGraph

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


test_graph = StateGraph(DataLoadState)