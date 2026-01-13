# research_workflow.py
"""Research workflow for company information gathering and filtering."""

# Standard library imports
import asyncio
import json
import logging
from typing import Any, Dict, cast

# Third-party imports
import dspy
from langgraph.graph import END, START, StateGraph

# Local imports
from job_writing_agent.agents.output_schema import (
    CompanyResearchDataSummarizationSchema,
)
from job_writing_agent.classes.classes import ResearchState
from job_writing_agent.tools.SearchTool import (
    TavilyResearchTool,
    filter_research_results_by_relevance,
)
from job_writing_agent.utils.llm_provider_factory import LLMFactory

logger = logging.getLogger(__name__)

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
QUERY_TIMEOUT = 30  # seconds
EVAL_TIMEOUT = 15  # seconds per evaluation


def validate_research_inputs(state: ResearchState) -> tuple[bool, str, str]:
    """
    Validate that required inputs are present in research state.

    Args:
        state: Current research workflow state

    Returns:
        Tuple of (is_valid, company_name, job_description)
    """
    try:
        # Safe dictionary access with fallbacks
        company_research_data = state.get("company_research_data", {})
        company_name = company_research_data.get("company_name", "")
        job_description = company_research_data.get("job_description", "")

        if not company_name or not company_name.strip():
            logger.error("Company name is missing or empty")
            return False, "", ""

        if not job_description or not job_description.strip():
            logger.error("Job description is missing or empty")
            return False, "", ""

        return True, company_name.strip(), job_description.strip()

    except (TypeError, AttributeError) as e:
        logger.error(f"Invalid state structure: {e}")
        return False, "", ""


def parse_dspy_queries_with_fallback(
    raw_queries: dict[str, Any], company_name: str
) -> dict[str, str]:
    """
    Parse DSPy query output with multiple fallback strategies.
    Returns a dict of query_id -> query_string.
    """
    try:
        # Try to extract search_queries field
        if isinstance(raw_queries, dict) and "search_queries" in raw_queries:
            queries_data = raw_queries["search_queries"]

            # If it's a JSON string, parse it
            if isinstance(queries_data, str):
                try:
                    queries_data = json.loads(queries_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode failed: {e}. Using fallback queries.")
                    return get_fallback_queries(company_name)

            # Extract query strings
            if isinstance(queries_data, dict):
                parsed = {}
                for key, value in queries_data.items():
                    if isinstance(value, str):
                        parsed[key] = value
                    elif isinstance(value, list) and len(value) > 0:
                        parsed[key] = str(value[0])

                if parsed:
                    return parsed

        # If we reach here, parsing failed
        logger.warning("Could not parse DSPy queries. Using fallback.")
        return get_fallback_queries(company_name)

    except Exception as e:
        logger.error(f"Error parsing DSPy queries: {e}. Using fallback.")
        return get_fallback_queries(company_name)


def get_fallback_queries(company_name: str) -> dict[str, str]:
    """
    Generate basic fallback queries when DSPy fails.
    """
    return {
        "query1": f"{company_name} company culture and values",
        "query2": f"{company_name} recent news and achievements",
        "query3": f"{company_name} mission statement and goals",
    }


def company_research_data_summary(state: ResearchState) -> ResearchState:
    """
    Summarize the filtered research data into a concise summary.

    Replaces the raw tavily_search results with a summarized version using LLM.

    Args:
        state: Current research state with search results

    Returns:
        Updated state with research summary
    """
    try:
        # Update current node
        updated_state = {**state, "current_node": "company_research_data_summary"}

        # Extract the current research data with safe access
        company_research_data = state.get("company_research_data", {})
        tavily_search_data = company_research_data.get("tavily_search", [])

        # If no research data, skip summarization
        if not tavily_search_data or len(tavily_search_data) == 0:
            logger.warning("No research data to summarize. Skipping summarization.")
            return updated_state

        logger.info(f"Summarizing {len(tavily_search_data)} research result sets...")

        # Create DSPy summarization chain
        company_research_data_summarization = dspy.ChainOfThought(
            CompanyResearchDataSummarizationSchema
        )

        # Initialize LLM provider

        llm_provider = LLMFactory()
        llm = llm_provider.create_dspy(
            model="mistralai/devstral-2512:free",
            provider="openrouter",
            temperature=0.3,
        )

        # Generate summary using DSPy
        with dspy.context(lm=llm, adapter=dspy.JSONAdapter()):
            response = company_research_data_summarization(
                company_research_data=company_research_data
            )
        # Extract the summary from the response with safe access
        summary_json_str = ""
        if hasattr(response, "company_research_data_summary"):
            summary_json_str = response.company_research_data_summary
        elif isinstance(response, dict):
            summary_json_str = response.get("company_research_data_summary", "")
        else:
            logger.error(
                f"Unexpected response format from summarization: {type(response)}"
            )
            return updated_state

        # Update state with summary using safe dictionary operations
        updated_company_research_data = {**company_research_data}
        updated_company_research_data["company_research_data_summary"] = (
            summary_json_str
        )
        updated_state["company_research_data"] = updated_company_research_data

        return updated_state

    except Exception as e:
        logger.error(f"Error in company_research_data_summary: {e}", exc_info=True)
        # Return state unchanged on error
        return updated_state


async def research_company_with_retry(state: ResearchState) -> ResearchState:
    """
    Research company with retry logic and timeouts.
    """
    state["current_node"] = "research_company"

    # Validate inputs
    is_valid, company_name, job_description = validate_research_inputs(state)

    if not is_valid:
        logger.error("Invalid inputs for research. Skipping research phase.")
        return ResearchState(
            company_research_data={
                **state.get("company_research_data", {}),
                "tavily_search": [],
            },
            attempted_search_queries=[],
            current_node="research_company",
            content_category=state.get("content_category", "cover_letter"),
            messages=state.get("messages", []),
        )

    logger.info(f"Researching company: {company_name}")

    # Try with retries
    for attempt in range(MAX_RETRIES):
        try:
            # Create tool instance
            tavily_search = TavilyResearchTool(
                job_description=job_description, company_name=company_name
            )

            # Generate queries with timeout
            queries_task = asyncio.create_task(
                asyncio.to_thread(tavily_search.create_tavily_queries)
            )

            try:
                raw_queries = await asyncio.wait_for(
                    queries_task, timeout=QUERY_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Query generation timed out (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                else:
                    raise

            # Parse queries with fallback
            # Convert DSPy Prediction to dict if needed
            if hasattr(raw_queries, "dict"):
                raw_queries_dict = cast(Dict[str, Any], raw_queries.dict())
            elif hasattr(raw_queries, "__dict__"):
                raw_queries_dict = cast(Dict[str, Any], raw_queries.__dict__)
            elif isinstance(raw_queries, dict):
                raw_queries_dict = cast(Dict[str, Any], raw_queries)
            else:
                raw_queries_dict = cast(Dict[str, Any], dict(raw_queries))

            queries = parse_dspy_queries_with_fallback(raw_queries_dict, company_name)

            if not queries:
                logger.warning("No valid queries generated")
                queries = get_fallback_queries(company_name)

            logger.info(
                f"Generated {len(queries)} search queries: {list(queries.keys())}"
            )

            # Perform searches with timeout
            search_task = asyncio.create_task(
                asyncio.to_thread(tavily_search.tavily_search_company, queries)
            )

            try:
                search_results = await asyncio.wait_for(
                    search_task, timeout=QUERY_TIMEOUT * len(queries)
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Search timed out (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                else:
                    raise

            # Validate results
            if not isinstance(search_results, list):
                logger.warning(f"Invalid search results type: {type(search_results)}")
                search_results = []

            if len(search_results) == 0:
                logger.warning("No search results returned")

            # Store results and return ResearchState
            return ResearchState(
                company_research_data={
                    **state.get("company_research_data", {}),
                    "tavily_search": search_results,
                },
                attempted_search_queries=list(queries.values()),
                current_node="research_company",
                content_category=state.get("content_category", "cover_letter"),
                messages=state.get("messages", []),
            )

        except Exception as e:
            logger.error(
                f"Error in research_company (attempt {attempt + 1}/{MAX_RETRIES}): {e}",
                exc_info=True,
            )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error("All retry attempts exhausted. Using empty results.")
                return ResearchState(
                    company_research_data={
                        **state.get("company_research_data", {}),
                        "tavily_search": [],
                    },
                    attempted_search_queries=[],
                    current_node="research_company",
                    content_category=state.get("content_category", "cover_letter"),
                    messages=state.get("messages", []),
                )

    return ResearchState(
        company_research_data=state.get("company_research_data", {}),
        attempted_search_queries=[],
        current_node="research_company",
        content_category=state.get("content_category", "cover_letter"),
        messages=state.get("messages", []),
    )


# Create research subgraph
research_subgraph = StateGraph(ResearchState)

# Add research subgraph nodes
research_subgraph.add_node("research_company", research_company_with_retry)
research_subgraph.add_node("relevance_filter", filter_research_results_by_relevance)
research_subgraph.add_node(
    "company_research_data_summary", company_research_data_summary
)

# Add research subgraph edges
research_subgraph.add_edge(START, "research_company")
research_subgraph.add_edge("research_company", "relevance_filter")
research_subgraph.add_edge("relevance_filter", "company_research_data_summary")
research_subgraph.add_edge("company_research_data_summary", END)

# Compile research subgraph
research_workflow = research_subgraph.compile()
