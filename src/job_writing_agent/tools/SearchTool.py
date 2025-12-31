import logging
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

from langchain_tavily import TavilySearch
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_HELPFULNESS_PROMPT
)
import dspy

from ..agents.output_schema import TavilySearchQueries
from ..classes.classes import ResearchState
from ..utils.llm_provider_factory import LLMFactory

logger = logging.getLogger(__name__)


env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)


openrouter_api_key = os.environ["OPENROUTER_API_KEY"]

llm_provider = LLMFactory()


class TavilyResearchTool:

    def __init__(self, job_description, company_name, max_results=5, model_name="qwen/qwen3-4b:free"):
        self.dspy_llm = llm_provider.create_dspy(model=model_name,
                                                 provider="openrouter",
                                                 temperature=0.3)
        self.job_description = job_description
        self.company_name = company_name
        self.tavily_searchtool  = TavilySearch(max_results=max_results)

    def create_tavily_queries(self):
        """
        Generate search queries for TavilySearch based on the job description and company name.
        Returns:
            dict: A dictionary containing the generated search queries.
        """
        tavily_query_generator = dspy.ChainOfThought(TavilySearchQueries)
        with dspy.context(lm=self.dspy_llm, adapter=dspy.JSONAdapter()):
            response = tavily_query_generator(job_description=self.job_description, company_name=self.company_name)
            return response


    def tavily_search_company(self, queries):
        
        query_results: list[list[str]] = []
        for query in queries:
            try:
                search_query_response = self.tavily_searchtool.invoke({"query": queries[query]})
                query_results.append([res['content'] for res in search_query_response['results']])
                # print(f"Tavily Search Tool Response for query '{search_query_response['query']}': {query_results_map[search_query_response['query']]}")
            except Exception as e:
                logger.error(f"Failed to perform company research using TavilySearchTool. Error : {e}")
                continue

        return query_results

llm_structured = llm_provider.create_langchain("llama3.1-8b",
                                                 provider="cerebras",
                                                 temperature=0.3)

def get_relevance_evaluator():
    return create_async_llm_as_judge(
                                    judge=llm_structured,
                                    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
                                    feedback_key="retrieval_relevance",
                                    )


def get_helpfulness_evaluator():
    return create_async_llm_as_judge(
                                    judge=llm_structured,
                                    prompt=RAG_HELPFULNESS_PROMPT
                                    + '\nReturn "true" if the answer is helpful, and "false" otherwise.',
                                    feedback_key="helpfulness",
                                    )


async def relevance_filter(state: ResearchState) -> ResearchState:
    try:
        # Set the current node
        state["current_node"] = "relevance_filter"

        # Get the all_query_data and attempted_queries_list
        tavily_search_results = state["company_research_data"]["tavily_search"]
        attempted_tavily_query_list = state["attempted_search_queries"]

        # Check if all_query_data and attempted_queries_list are lists
        assert isinstance(tavily_search_results, list), "tavily_search_results is not a list"
        assert isinstance(attempted_tavily_query_list, list), "attempted_tavily_query_list is not a list"

        print("Filtering results...")

        filtered_search_results = []  # Stores results deemed relevant in this specific call

        # Create a semaphore to limit concurrent tasks to 2
        semaphore = asyncio.Semaphore(2)

        async def evaluate_with_semaphore(query_result_item, input_query: str):
            # query_result_item is a dict like {'rationale': '...', 'results': [...]}
            async with semaphore:
                relevance_evaluator = get_relevance_evaluator()
                eval_result = await relevance_evaluator(
                    inputs=input_query, context=query_result_item  # context is the whole result block for the query
                )
                return query_result_item, eval_result

        # Create tasks for all results
        tasks: list = []

        for query_result, attempted_query in zip(tavily_search_results, attempted_tavily_query_list):
            tasks.append(evaluate_with_semaphore(query_result, attempted_query))
        # Process tasks as they complete
        for completed_task in asyncio.as_completed(tasks):
            query_result_item, eval_result = await completed_task
            # logger.info(f"Evaluated query result for '{query_result_item}': {eval_result}")
            if eval_result.get("score"):  # Safely check for score
                if isinstance(query_result_item, list):
                    filtered_search_results.extend(query_result_item)
                else:
                    # Handle cases where "results" might not be a list or is missing
                    logger.warning("Expected a list in query_result_item, got: %s", type(query_result_item))

        # Append the newly filtered results to the main compiled_results list
        state["company_research_data"]["tavily_search"] = filtered_search_results

        logger.info(f"Relevance filtering completed. {len(filtered_search_results)} relevant results found.")

        return state

    except Exception as e:
        print(f"ERROR in relevance_filter: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error in relevance_filter: {str(e)}")
        # Return original state to avoid breaking the flow
        return state
