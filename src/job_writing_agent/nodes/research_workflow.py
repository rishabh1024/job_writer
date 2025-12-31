# -*- coding: utf-8 -*-
"""
This module performs the research phase of the job application writing process.
One of the stages is Tavily Search which will be use to search for the company
"""

import logging
import json
from langgraph.graph import StateGraph, START, END

from job_writing_agent.tools.SearchTool import TavilyResearchTool
from job_writing_agent.classes.classes import ResearchState
from job_writing_agent.tools.SearchTool import relevance_filter


logger = logging.getLogger(__name__)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def research_company(state: ResearchState) -> ResearchState:
    """Research the company if name is available."""
    state["current_node"] = "research_company"

    try:
        # Extract values from state
        company_name: str = state["company_research_data"].get("company_name", None)
        job_description = state["company_research_data"].get("job_description", None)

        assert company_name is not None, "Company name is required for research_company"
        assert job_description is not None, (
            "Job description is required for research_company"
        )

        logger.info(f"Researching company: {company_name}")

        # Call search_company using the invoke method instead of __call__
        # The tool expects job_description and company_name and returns a tuple
        tavily_search = TavilyResearchTool(
            job_description=job_description, company_name=company_name
        )

        tavily_search_queries = tavily_search.create_tavily_queries()

        tavily_search_queries_json: dict = json.loads(
            tavily_search_queries["search_queries"]
        )

        logger.info(list(tavily_search_queries_json.values()))

        tavily_search_results: list[list[str]] = tavily_search.tavily_search_company(
            tavily_search_queries_json
        )

        assert isinstance(tavily_search_results, list), (
            "Expected list or tuple from tavily_search_company"
        )
        assert len(tavily_search_results) > 0, (
            "No results returned from tavily_search_company"
        )
        assert len(tavily_search_queries_json) > 0, "No search queries were attempted"

        logger.info(
            f"Search completed with results and {len(tavily_search_queries)} queries"
        )

        # Store results in state - note that results is the first item in the tuple
        state["attempted_search_queries"] = list(tavily_search_queries_json.values())
        state["company_research_data"]["tavily_search"] = tavily_search_results

    except Exception as e:
        logger.error(f"Error in research_company: {str(e)}")
        # Provide empty results to avoid breaking the workflow
        state["company_research_data"]["tavily_search"] = []
        state["attempted_search_queries"] = []
    finally:
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
