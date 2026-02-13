# Standard library imports
import asyncio
import logging
import os
from pathlib import Path

# Third-party imports
import dspy
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import RAG_HELPFULNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT

# Local imports
from ..agents.output_schema import TavilySearchQueries
from ..classes.classes import ResearchState
from ..utils.llm_provider_factory import LLMFactory


logger = logging.getLogger(__name__)


env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


# Safe environment variable access with validation
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    logger.error("OPENROUTER_API_KEY environment variable not set")
    raise ValueError("OPENROUTER_API_KEY environment variable is required")


class TavilyResearchTool:
    def __init__(
        self,
        job_description,
        company_name,
        max_results=5,
        model_name="google/gemma-3-2google/gemma-3-27b-it:free7b-it:free",
    ):
        # Create LLM inside __init__ (lazy initialization)
        llm_provider = LLMFactory()
        self.dspy_llm = llm_provider.create_dspy(
            model=model_name, provider="openrouter", temperature=0.3
        )
        self.job_description = job_description
        self.company_name = company_name
        self.tavily_searchtool = TavilySearch(max_results=max_results)

    def create_tavily_queries(self):
        """
        Generate search queries for TavilySearch based on the job description and company name.
        Returns:
            dict: A dictionary containing the generated search queries.
        """
        tavily_query_generator = dspy.ChainOfThought(TavilySearchQueries)
        with dspy.context(lm=self.dspy_llm, adapter=dspy.JSONAdapter()):
            response = tavily_query_generator(
                job_description=self.job_description, company_name=self.company_name
            )
            return response

    def tavily_search_company(self, queries):
        """
        Execute Tavily searches for multiple queries.

        Args:
            queries: Dictionary of query identifiers to query strings

        Returns:
            List of search result lists, one per query
        """
        query_results: list[list[str]] = []
        for query_key in queries:
            try:
                query_string = queries.get(query_key, "")
                if not query_string:
                    logger.warning(f"Empty query for key: {query_key}")
                    continue

                search_query_response = self.tavily_searchtool.invoke(
                    {"query": query_string}
                )
                # Safe dictionary access for response
                results = search_query_response.get("results", [])
                query_results.append(
                    [res.get("content", "") for res in results if isinstance(res, dict)]
                )
            except Exception as e:
                logger.error(
                    f"Failed to perform company research using TavilySearchTool. Error: {e}"
                )
                continue

        return query_results


def get_relevance_evaluator():
    """
    Create an LLM-as-judge evaluator for relevance filtering.

    Creates the LLM on-demand (lazy initialization) to avoid startup delays.
    """
    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm_structured = llm_provider.create_langchain(
        "llama3.1-8b", provider="cerebras", temperature=0.3
    )
    return create_async_llm_as_judge(
        judge=llm_structured,
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        feedback_key="retrieval_relevance",
    )


def get_helpfulness_evaluator():
    """
    Create an LLM-as-judge evaluator for helpfulness filtering.

    Creates the LLM on-demand (lazy initialization) to avoid startup delays.
    """
    # Create LLM inside function (lazy initialization)
    llm_provider = LLMFactory()
    llm_structured = llm_provider.create_langchain(
        "llama3.1-8b", provider="cerebras", temperature=0.3
    )
    return create_async_llm_as_judge(
        judge=llm_structured,
        prompt=RAG_HELPFULNESS_PROMPT
        + '\nReturn "true" if the answer is helpful, and "false" otherwise.',
        feedback_key="helpfulness",
    )


async def filter_research_results_by_relevance(state: ResearchState) -> ResearchState:
    """
    Filter search results to keep only relevant company information.
    Uses LLM-as-judge to evaluate if each result set is relevant to its query.
    Irrelevant results are REMOVED from the final output.
    """
    try:
        state["current_node"] = "filter_research_results_by_relevance"

        # Extract and validate required state fields once
        company_research_data = state.get("company_research_data", {})
        raw_search_results = company_research_data.get("tavily_search", [])
        search_queries_used = state.get("attempted_search_queries", [])

        # Validate data types
        if not isinstance(raw_search_results, list):
            logger.warning(f"Invalid search results type: {type(raw_search_results)}")
            return state

        if not isinstance(search_queries_used, list):
            logger.warning(f"Invalid queries type: {type(search_queries_used)}")
            search_queries_used = []

        # Early exit if no results
        if len(raw_search_results) == 0:
            logger.info("No search results to filter.")
            # Update using the extracted variable
            company_research_data["tavily_search"] = []
            state["company_research_data"] = company_research_data
            return state

        logger.info(
            f"Starting relevance filtering for {len(raw_search_results)} result sets..."
        )

        # Track filtering statistics
        results_kept = []
        results_removed_count = 0
        evaluation_errors_count = 0

        # Limit concurrent evaluations to prevent rate limiting
        concurrency_limiter = asyncio.Semaphore(2)

        async def evaluate_result_set_relevance(
            search_result_content, original_query: str
        ):
            """
            Evaluate if a search result set is relevant to its query.

            Returns:
                tuple: (search_result_content, is_relevant: bool, error: str|None)
            """
            async with concurrency_limiter:
                try:
                    # Skip empty result sets
                    if not search_result_content:
                        logger.debug(
                            f"Skipping empty result set for query: {original_query[:50]}..."
                        )
                        return (None, False, "empty")

                    # Create relevance evaluator
                    llm_relevance_judge = get_relevance_evaluator()

                    # Evaluate with timeout protection
                    evaluation_task = llm_relevance_judge(
                        inputs=original_query, context=search_result_content
                    )

                    evaluation_result = await asyncio.wait_for(
                        evaluation_task, timeout=15
                    )

                    # Extract relevance score (True = relevant, False = not relevant)
                    is_result_relevant = bool(evaluation_result.get("score", False))

                    if is_result_relevant:
                        logger.debug(
                            f"KEPT: Result relevant for query: {original_query[:60]}..."
                        )
                        return (search_result_content, True, None)
                    else:
                        logger.debug(
                            f"REMOVED: Result not relevant for query: {original_query[:60]}..."
                        )
                        return (None, False, None)

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Evaluation timed out for query: {original_query[:60]}... (KEEPING result)"
                    )
                    # Keep the result on timeout to avoid losing potentially useful data
                    return (search_result_content, True, "timeout")

                except Exception as e:
                    logger.error(
                        f"Evaluation failed for query: {original_query[:60]}... - {e} (KEEPING result)"
                    )
                    return (search_result_content, True, f"error:{str(e)}")

        # Create evaluation tasks for all result sets
        evaluation_tasks = []
        for result_set, query in zip(raw_search_results, search_queries_used):
            task = evaluate_result_set_relevance(result_set, query)
            evaluation_tasks.append(task)

        # Execute all evaluations concurrently
        all_evaluation_results = await asyncio.gather(
            *evaluation_tasks, return_exceptions=True
        )

        # Process evaluation results and separate kept vs removed
        for eval_result in all_evaluation_results:
            # Handle exceptions from gather
            if isinstance(eval_result, Exception):
                logger.error(f"Evaluation task failed with exception: {eval_result}")
                evaluation_errors_count += 1
                continue

            # Type guard: eval_result is now guaranteed to be a tuple
            if not isinstance(eval_result, tuple) or len(eval_result) != 3:
                logger.error(
                    f"Unexpected evaluation result format: {type(eval_result)}"
                )
                evaluation_errors_count += 1
                continue

            result_content, is_relevant, error = eval_result

            # Track errors
            if error:
                evaluation_errors_count += 1

            # Keep relevant results, discard irrelevant ones
            if result_content is not None and is_relevant:
                results_kept.append(result_content)
            else:
                results_removed_count += 1

        # Update company_research_data with ONLY the relevant results
        company_research_data["tavily_search"] = results_kept
        state["company_research_data"] = company_research_data

        # Log filtering summary
        total_evaluated = len(raw_search_results)
        kept_count = len(results_kept)
        removed_count = results_removed_count

        logger.info(
            f"Relevance filtering complete: "
            f"KEPT {kept_count} | REMOVED {removed_count} | TOTAL {total_evaluated} "
            f"({evaluation_errors_count} evaluation errors)"
        )

        return state

    except Exception as e:
        logger.error(f"Critical error in relevance filtering: {e}", exc_info=True)
        # On critical error, return original state unchanged
        return state
