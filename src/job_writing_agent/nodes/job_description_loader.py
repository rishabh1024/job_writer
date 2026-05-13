# -*- coding: utf-8 -*-
"""
Job Description Loader Module

This module provides the JobDescriptionLoader class responsible for loading and parsing
job description files and URLs, extracting both the job posting text and company name.
"""

import logging
from typing import Any, Awaitable, Callable, ClassVar, Optional, Tuple

from deprecated import deprecated
from langchain_core.documents import Document

from job_writing_agent.utils.document_processing import parse_job_description
from job_writing_agent.utils.logging.logging_decorators import (
    log_async,
    log_errors,
)

logger = logging.getLogger(__name__)


class JobDescriptionLoader:
    """
    Responsible for loading and parsing job description documents.

    This class follows SOLID principles:
    - Single Responsibility: Only handles job description parsing
    - Dependency Inversion: Parser is injected for testability
    - Open/Closed: Can extend with different parsers without modification
    - Interface Segregation: Focused interface (only job description methods)

    Example:
        >>> loader = JobDescriptionLoader()
        >>> job_text, company = await loader.load_job_description("https://example.com/job")
        >>>
        >>> # With custom parser for testing
        >>> async def mock_parser(source):
        ...     return Document(page_content="test", metadata={"company_name": "TestCo"})
        >>> loader = JobDescriptionLoader(parser=mock_parser)
    """

    company_name: ClassVar[str]
    job_posting_text: ClassVar[str]

    def __init__(
        self, parser: Optional[Callable[[Any], Awaitable[Document]]] = None
    ):
        """
        Initialize JobDescriptionLoader with optional parser dependency injection.

        Parameters
        ----------
        parser: Optional[Callable[[Any], Awaitable[Document]]]
            Async function to parse job description from URL. Defaults to
            `parse_job_description` from document_processing. Can be injected
            for testing.

            The parser should take a URL (str) and return an awaitable that
            resolves to a Document with page_content (str) and metadata (dict).
        """
        self._parser = parser or parse_job_description

    @log_async
    @log_errors
    async def load_job_description(self, job_description_url: Any) -> Tuple[str, str]:
        """
        Load a job description from a URL and return its text and company name.

        Callers (e.g. load_job_description_node) typically validate that the URL
        is non-empty before calling. Company name is read from document metadata.

        Parameters
        ----------
        job_description_url: Any
            URL of the job posting (http:// or https://).

        Returns
        -------
        Tuple[str, str]
            (job_posting_text, company_name). Empty strings if not found.

        Raises
        ------
        Exception
            If parsing or fetching fails.
        """

        logger.info("Loading job description from: %s", job_description_url)

        job_description_document: Document = await self._parser(job_description_url)

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

    @log_async
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

    @deprecated(
        version="1.0.0",
        reason="Job description prompting now uses LangGraph interrupts (prompt_user_for_job_description_node). "
        "This method used synchronous input() which blocked async execution and was not suitable for web deployment.",
    )
    async def _prompt_user_for_job_description(self) -> str:
        """
        Prompt the user for job description text via stdin.

        Kept for backward compatibility only. Use prompt_user_for_job_description_node
        with LangGraph interrupt in new code.

        Returns
        -------
        str
            User input string.
        """
        return input("Please paste the job description in text format: ")
