import logging
import os
import json
import asyncio


from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import tool
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_HELPFULNESS_PROMPT
)

from ..utils.llm_client import LLMClient
from ..agents.output_schema import TavilyQuerySet
from ..prompts.templates import TAVILY_QUERY_PROMPT
from ..classes.classes import ResearchState

logger = logging.getLogger(__name__)

LLM = LLMClient()
llm_client = LLM.get_instance(model_name="ejschwar/llama3.2-better-prompts:latest", model_provider="ollama_llm")
llm_structured = llm_client.get_llm()

relevance_evaluator = create_async_llm_as_judge(
    judge=llm_structured,
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
)

helpfulness_evaluator = create_async_llm_as_judge(
    judge=llm_structured,
    prompt=RAG_HELPFULNESS_PROMPT
    + '\nReturn "true" if the answer is helpful, and "false" otherwise.',
    feedback_key="helpfulness",
)

@tool
def search_company(job_description: str, company_name: str) -> dict:
    """Gather information about a company to understand more about the role,
    recent developments, culture, and values of the company."""

    try:
        # Get format instructions from the parser
        base_parser = PydanticOutputParser(pydantic_object=TavilyQuerySet)
        parser = RetryOutputParser.from_llm(llm_structured, base_parser)
        format_instructions = parser.get_format_instructions()


        # Create the prompt with both messages
        chat_prompt_tavily: ChatPromptTemplate = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                TAVILY_QUERY_PROMPT,
                input_variables=["company_name"]
            ),
            HumanMessagePromptTemplate.from_template(
                "Below is the required job description to parse:\n\n{job_description}",
                input_variables=["job_description"]
            )
        ])


        chat_prompt_value: PromptValue = chat_prompt_tavily.format_prompt(
            company_name=company_name,
            job_description=job_description
        )


        # Format messages and get LLM response
        chat_prompt_tavily_messages = chat_prompt_tavily.format_messages(
            company_name=company_name,
            job_description=job_description
        )
        

        # Get response from LLM
        search_results_llm = llm_structured.invoke(chat_prompt_tavily_messages)
        # logger.info("Raw LLM Response content: %s", search_results_llm.content)


        try:
            parsed_query_set: TavilyQuerySet = parser.parse_with_prompt(search_results_llm.content, chat_prompt_value)
            logger.info("Parsed TavilyQuerySet: %s", parsed_query_set.model_dump_json(indent=2))
        except json.JSONDecodeError as e:
            logger.error("JSON decoding error while parsing LLM response: %s. LLM content was: %s", e, search_results_llm.content, exc_info=True)
            raise
        except Exception as e: # Catches PydanticValidationErrors and other parsing issues
            logger.error("Error parsing TavilyQuerySet from LLM completion: %s. LLM content was: %s", e, search_results_llm.content, exc_info=True)
            raise


        # Initialize search with advanced parameters
        search = TavilySearchResults(max_results=4, search_depth="advanced")

        
        # Prepare the structure for storing queries, rationales, and Tavily results
        company_research_data = {}
        attempted_queries = []
        query_attributes = [f"query{i}" for i in range(1, 6)]


        for attr_name in query_attributes:
            query_list = getattr(parsed_query_set, attr_name, None)
            if query_list and isinstance(query_list, list) and len(query_list) > 0:
                actual_query = query_list[0]
                rationale = query_list[1] if len(query_list) > 1 else "N/A" # Handle if rationale is missing
                company_research_data[attr_name] = {
                    'query': actual_query,
                    'rationale': rationale,
                    'results': []
                }


        # logger.info("Prepared company research structure: %s", json.dumps(company_research_data, indent=2))
        # Execute each query and store results
        for query_key, query_info in company_research_data.items():
            try:
                if not isinstance(query_info['query'], str) or not query_info['query'].strip():
                    logger.warning("Skipping Tavily search for %s due to invalid/empty query: '%s'", query_key, query_info['query'])
                    query_info['results'] = []
                    continue


                logger.info("Executing Tavily search for %s: '%s'", query_key, query_info['query'])
                # tool.invoke({"args": {'query': 'who won the last french open'}, "type": "tool_call", "id": "foo", "name": "tavily"})
                tavily_api_results = search.invoke({"args": {'query': query_info['query']}, "type": "tool_call", "id": "job_search", "name": "tavily"})
                attempted_queries.append(query_info['query'])
                del query_info['query']
                
                if tavily_api_results and isinstance(tavily_api_results, list) and len(tavily_api_results) > 0:
                    query_info['results'] = [result['content'] for result in tavily_api_results if 'content' in result]
                else:
                    logger.info("No results or unexpected format from Tavily for %s.", query_key)
                    query_info['results'] = []
            except Exception as e:
                logger.error("Error executing Tavily search for query %s ('%s'): %s", query_key, query_info['query'], str(e), exc_info=True)
                query_info['results'] = []
        
        # print("Results: ", results)
        return company_research_data, attempted_queries

    except json.JSONDecodeError as e:
        logger.error("JSON decoding error: %s", e)
        raise
    except AttributeError as e:
        logger.error("Attribute error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

async def relevance_filter(state: ResearchState) -> ResearchState:
    try:
        # Mark the current node
        state["current_node"] = "relevance_filter"
        
        # Check if company_research_data exists
        if not state.get("company_research_data"):
            print("ERROR: company_research_data not found in state")
            return state
            
        # Check if tavily_search results exist
        if not state["company_research_data"].get("tavily_search"):
            print("ERROR: tavily_search not found in company_research_data")
            state["company_research_data"]["tavily_search"] = []
            return state
        
        # Initialize compiled_results if not present
        if "compiled_results" not in state:
            state["compiled_results"] = []
            
        print("Filtering results...")
        # Get the company research data which contains results for different queries
        # Example: {'query1': {'rationale': ..., 'results': [...]}, 'query2': ...}

        all_query_data = state["company_research_data"].get("tavily_search", {})
        # print("All query data:", all_query_data)
        filtered_results_for_current_run = [] # Stores results deemed relevant in this specific call

        # Create a semaphore to limit concurrent tasks to 2
        semaphore = asyncio.Semaphore(2)

        async def evaluate_with_semaphore(query_result_item: dict):
            # query_result_item is a dict like {'rationale': '...', 'results': [...]}
            async with semaphore:
                # Safely get the query to use for relevance evaluation
                attempted_queries_list = state.get("attempted_search_queries", [])
                input_query = attempted_queries_list[-1] if attempted_queries_list else "No query context available"

                eval_result = await relevance_evaluator(
                    inputs=input_query, context=query_result_item  # context is the whole result block for the query
                )
                return query_result_item, eval_result

        # Create tasks for all results
        tasks = [evaluate_with_semaphore(query_info) for query_info in all_query_data.values() if isinstance(query_info, dict) and "results" in query_info]

        # Process tasks as they complete
        for completed_task in asyncio.as_completed(tasks):
            query_result_item, eval_result = await completed_task
            if eval_result.get("score"): # Safely check for score
                # Assuming query_result_item["results"] is a list of content strings
                if isinstance(query_result_item.get("results"), list):
                    # print(f"Evaluated result: {query_result_item}")
                    filtered_results_for_current_run.extend(query_result_item["results"])
                else:
                    # Handle cases where "results" might not be a list or is missing
                    logger.warning("Expected a list for 'results' in query_result_item, got: %s", type(query_result_item.get('results')))

        logger.info("Filtered results for current run: %s",filtered_results_for_current_run)
        
        # The error occurs at a line like the following (line 178 in your traceback):
        # This print statement will now safely access "compiled_results"
        # print("Compiled results (before append): ", state["compiled_results"])    # Append the newly filtered results to the main compiled_results list
        state["compiled_results"].extend(filtered_results_for_current_run)
        state["company_research_data"]["tavily_search"] = filtered_results_for_current_run
        # logger.info(f"Compiled results (after append): {state['compiled_results']}")
        return state
    
    except Exception as e:
        print(f"ERROR in relevance_filter: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error in relevance_filter: {str(e)}")
        # Return original state to avoid breaking the flow
        return state