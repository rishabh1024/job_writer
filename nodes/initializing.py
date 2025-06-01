# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:49:52 2023
@author: rishabhaggarwal
"""
import os
import logging
from typing_extensions import Literal

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage

from job_writer.classes import AppState, DataLoadState
from job_writer.utils.document_processing import (
    parse_resume,
    get_job_description
)


logger = logging.getLogger(__name__)


class Dataloading:
    """
    Initialize the state for the job application writer workflow.
    """
    def __init__(self):
        pass
   

    async def system_setup(self, state: AppState) -> DataLoadState:
        """Initialize conversation by setting up a persona through System Prompt."""

        resume_path = state.get("resume_path")

        # Verify if the resume file path provided is valid
        if not resume_path:
            logger.error("Resume path is not provided in the state.")
        elif not os.path.exists(resume_path):
            logger.error("Resume file does not exist at path: %s", resume_path)
            # Similar handling as above:
            # raise FileNotFoundError(f"Resume file not found: {resume_path}")
        elif not os.path.isfile(resume_path):
            logger.error("The path provided for the resume is not a file: %s", resume_path)
            # Similar handling:
            # raise ValueError(f"Resume path is not a file: {resume_path}")
        else:
            logger.info("Resume path verified: %s", resume_path)


        persona_init_message = SystemMessage(
            content="You are my dedicated assistant for writing job application content, "
                   "including cover letters, LinkedIn outreach messages, and responses to "
                   "job-specfific questions (e.g., experience, culture fit, or motivation)."
        )
        messages = state.get("messages", [])
        messages.append(persona_init_message)

        return {
            **state,
            "messages": messages,
            "current_node": "initialize_system"

        }


    async def get_resume(self, resume_source):
        """
        Get the resume te
        """
        try:
            print("Parsing resume....")
            resume_text = ""
            resume_chunks = parse_resume(resume_source)
            for chunk in resume_chunks:
                if hasattr(chunk, 'page_content') and chunk.page_content:
                        resume_text += chunk.page_content
                elif isinstance(chunk, str) and chunk: # If parse_resume (util) returns list of strings
                     resume_text += chunk
                else:
                    logger.debug("Skipping empty or invalid chunk in resume: %s", chunk)
                continue
            return resume_text
        except Exception as e:
            print(f"Error parsing resume: {e}")
            raise e
    

    async def parse_job_description(self, job_description_source):
        try:
            logger.info("Parsing job description from: %s", job_description_source)
            document: Document = get_job_description(job_description_source)

            company_name = ""
            job_posting_text = ""

            if document:
                # Extract company name from metadata
                if hasattr(document, 'metadata') and isinstance(document.metadata, dict):
                    company_name = document.metadata.get("company_name", "")
                    if not company_name:
                        logger.warning("Company name not found in job description metadata.")
                else:
                    logger.warning("Metadata attribute not found or not a dictionary in the Document for job description.")

                # Extract the job posting text from page_content
                if hasattr(document, 'page_content'):
                    job_posting_text = document.page_content
                    if not job_posting_text:
                        logger.info("Parsed job posting text is empty.")
                else:
                    logger.warning("page_content attribute not found in the Document for job description.")
            else:
                logger.warning("get_job_description returned None for source: %s", job_description_source)
            
            return job_posting_text, company_name

        except Exception as e:
            logger.error("Error parsing job description from source '%s': %s", job_description_source, e, exc_info=True)
            raise e

    async def load_inputs(self, state: DataLoadState) -> AppState:
        """
        Parse the resume and job description to prepare the data from the context 
        which is required for the job application writer for the current state
        """
        
        resume_source = state.get("resume_path", "")
        job_description_source = state.get("job_description_source", None)

        # Initialize result containers\
        resume_text = ""
        job_posting_text = ""
        company_name = ""
        resume_chunks = []        # Handle job description input
        if job_description_source:
            try:
                job_posting_text, company_name = await self.parse_job_description(job_description_source)
                print(f"Job description parsing complete. Length: {len(job_posting_text) if job_posting_text else 0}")
                
                # Ensure job_posting_text is not empty
                if not job_posting_text:
                    print("WARNING: Job posting text is empty after parsing.")
                    job_posting_text = "No job description available. Please check the URL or provide a different source."
            except Exception as e:
                print(f"Error parsing job description: {e} in file {__file__}")
                # Set a default value to prevent errors
                job_posting_text = "Error parsing job description."
                company_name = "Unknown Company"

        if resume_source:
            try:
                resume_text = await self.get_resume(resume_source)
            except Exception as e:
                print(f"Error parsing resume: {e} in file {__file__}")
                raise e


        # If either is missing, prompt the user
        if state["current_node"] == "verify" and not resume_text:
            resume_chunks = input("Please paste the resume in text format: ")
            resume_text = [Document(page_content=resume_chunks, metadata={"source": "resume"})]


        if state["current_node"] == "verify" and not job_posting_text:
            job_text = input("Please paste the job posting in text format: ")
            job_posting_text = [job_text]
                
            
        # Extract company name
        state["company_research_data"] = {'resume': resume_text, 'job_description': job_posting_text, 'company_name': company_name}

        state["current_node"] = "load_inputs"
        
        return state
    

    def validate_data_load_state(self,state: DataLoadState):
        assert state.company_research_data.get("resume"), "Resume is missing in company_research_data"
        assert state.company_research_data.get("job_description"), "Job description is missing"


    def verify_inputs(self, state: AppState) -> Literal["load", "research"]:
        """Verify that required inputs are present."""
        
        print("Verifying Inputs")
        state["current_node"] = "verify"

        logger.info("Verifying loaded inputs!")

        assert state["company_research_data"].get("resume"), "Resume is missing in company_research_data"
        assert state["company_research_data"].get("job_description"), "Job description is missing"

        if not state.get("company_research_data"):
            missing_items = []
            if not state.get("company_research_data").get("resume", ""):
                missing_items.append("resume")
            if not state.get("company_research_data").get("job_description", ""):
                missing_items.append("job description")
            print(f'Missing required data: {", ".join(missing_items)}')

            return "load"
                
        # Normalize state content to strings
        for key in ["resume", "job_description"]:
            try:
                if isinstance(state["company_research_data"][key], (list, tuple)):
                    state["company_research_data"][key] = " ".join(str(x) for x in state["company_research_data"][key])
                elif isinstance(state["company_research_data"][key], dict):
                    state["company_research_data"][key] = str(state["company_research_data"][key])
                else:
                    state["company_research_data"][key] = str(state["company_research_data"][key])
            except Exception as e:
                logger.warning("Error converting %s to string: %s", key, e)
                raise e
                
        return "research"

    async def run(self, state: DataLoadState) -> AppState:
        """
        Run the InitializeState class to initialize
        the state for the job application writer workflow.
        """
        state = await self.load_inputs(state)
        return state