# -*- coding: utf-8 -*-
"""
This module performs the research phase of the job application writing process.
One of the stages is Tavily Search which will be use to search for the company
"""
import logging
from langgraph.graph import StateGraph, START, END

from job_writer.tools.TavilySearch import relevance_filter, search_company
from job_writer.classes.classes import ResearchState

logger = logging.getLogger(__name__)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def research_company(state: ResearchState) -> ResearchState:
    """Research the company if name is available."""
    state["current_node"] = "research_company"
    
    try:
        # Extract values from state
        company_name = state["company_research_data"].get("company_name", "")
        job_description = state["company_research_data"].get("job_description", "")
        
        logger.info(f"Researching company: {company_name}")
        # Call search_company using the invoke method instead of __call__
        # The tool expects job_description and company_name and returns a tuple
        result = search_company.invoke({
            "job_description": job_description,
            "company_name": company_name
        })
        # Unpack the tuple
        if isinstance(result, tuple) and len(result) == 2:
            results, attempted_tavily_query_list = result
        else:
            # Handle the case when it's not a tuple
            results = result
            attempted_tavily_query_list = []
        
        logger.info(f"Search completed with results and {len(attempted_tavily_query_list)} queries")
        
        # Store results in state - note that results is the first item in the tuple
        state["attempted_search_queries"] = attempted_tavily_query_list
        state["company_research_data"]["tavily_search"] = results
        
    except Exception as e:
        logger.error(f"Error in research_company: {str(e)}")
        # Provide empty results to avoid breaking the workflow
        state["company_research_data"]["tavily_search"] = {"error": str(e), "tavily_search": []}
        state["attempted_search_queries"] = []
        
    return state

print("\n\n\nInitializing research workflow...\n\n\n")
# Create research subgraph
research_subgraph = StateGraph(ResearchState)

# Add research subgraph nodes
research_subgraph.add_node("research_company", research_company)
research_subgraph.add_node("relevance_filter", relevance_filter)


# Add research subgraph edges
research_subgraph.add_edge(START, "research_company")
research_subgraph.add_edge("research_company", "relevance_filter")
research_subgraph.add_edge("relevance_filter", END)

# Compile research subgraph
research_workflow = research_subgraph.compile()


# class ResearchWorkflow:
    
#     def __init__(self):
#         self.research_workflow = research_workflow
    


