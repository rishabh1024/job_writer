# -*- coding: utf-8 -*-
"""
Job Description Loader Module

This module provides the JobDescriptionLoader class responsible for loading and parsing
job description files and URLs, extracting both the job posting text and company name.
"""

import logging
from typing import Any, Awaitable, Callable, ClassVar, Optional, Tuple

from langchain_core.documents import Document

from job_writing_agent.utils.document_processing import (
    parse_job_description_from_url,
)

logger = logging.getLogger(__name__)


class JobDescriptionLoader:
    """
    Responsible for loading and parsing job
    description from website links or file uploads
    """

    company_name: ClassVar[str]
    job_posting_text: ClassVar[str]

    def __init__(
        self, parser: Optional[Callable[[Any], Awaitable[Document]]] = None
    ):
        """
        Initialize JobDescriptionLoader with optional parser dependency injection.
        """
        self._parser = parser or parse_job_description_from_url

    async def parse_job_description(
        self, job_description_source: Any
    ) -> Tuple[str, str]:
        """
        Parse a job description and return its text and company name.

        Extracts both the job posting text and company name from the document.
        Company name is extracted from document metadata if available.
        """

        logger.info("Parsing job description from: %s", job_description_source)
        assert job_description_source is not None, (
            "Job description source cannot be None"
        )

        job_description_document: Document = await self._parser(
            job_description_source
        )

        company_name = ""
        job_posting_text = ""

        # Extract company name from metadata
        if hasattr(job_description_document, "metadata") and isinstance(
            job_description_document.metadata, dict
        ):
            company_name = job_description_document.metadata.get(
                "company_name", ""
            )
            if not company_name:
                logger.warning(
                    "Company name not found in job description metadata."
                )
        else:
            logger.warning(
                "Metadata attribute missing or not a dict in job description document."
            )

        # Extract job posting text
        if hasattr(job_description_document, "page_content"):
            job_posting_text = job_description_document.page_content or ""
            if not job_posting_text:
                logger.info("Parsed job posting text is empty.")
        else:
            logger.warning(
                "page_content attribute missing in job description document."
            )

        return job_posting_text, company_name

    async def _load_job_description(self, jd_source: Any) -> Tuple[str, str]:
        """
        Load job description text and company name, raising if missing.

        This is a wrapper around parse_job_description() that validates the
        source first. Used by subgraph nodes for consistent error handling.

        Parameters
        ----------
        jd_source: Any
            Source for the job description (URL, file path, etc.).

        Returns
        -------
        Tuple[str, str]
            A tuple of (job_posting_text, company_name).

        Raises
        ------
        ValueError
            If jd_source is None or empty.
        """
        if not jd_source:
            raise ValueError("job_description_source is required")
        return await self.parse_job_description(jd_source)

    async def get_application_form_details(self, job_description_source: Any):
        """
        Placeholder for future method to get application form details.

        This method will be implemented to extract form fields and requirements
        from job application forms.

        Parameters
        ----------
        job_description_source: Any
            Source of the job description or application form.
        """
        # TODO: Implement form field extraction
        pass

    async def _prompt_user_for_job_description(self) -> str:
        """
        Prompt the user for input

        Parameters
        ----------
        prompt_msg: str
            Message to display to the user.

        Returns
        -------
        str
            User input string.
        """
        # In a real async UI replace input with an async call.
        return input("Please paste the job description in text format: ")
