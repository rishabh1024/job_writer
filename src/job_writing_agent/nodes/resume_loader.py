# -*- coding: utf-8 -*-
"""
Resume Loader Module

This module provides the ResumeLoader class responsible for loading and parsing
the resume file and returning the resume in the required format.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from job_writing_agent.utils.document_processing import (
    get_resume as get_resume_docs,
    parse_resume,
)
from job_writing_agent.utils.logging.logging_decorators import (
    log_async,
    log_errors,
)

logger = logging.getLogger(__name__)


class ResumeLoader:
    """
    Responsible for loading and parsing resume documents.

    Example:
        >>> loader = ResumeLoader()
        >>> resume_text = await loader.get_resume("path/to/resume.pdf")
        >>>
        >>> # With custom parser for testing
        >>> mock_parser = lambda x: [Document(page_content="test")]
        >>> loader = ResumeLoader(parser=mock_parser)
    """

    def __init__(self, parser: Optional[Callable[[Any], Any]] = None):
        """
        Initialize ResumeLoader with optional parser dependency injection.

        Parameters
        ----------
        parser: Optional[Callable[[Any], Any]]
            Function to parse resume documents. Defaults to `parse_resume` from
            document_processing. Can be injected for testing or custom parsing.
        """
        self._parser = parser or parse_resume

    @log_async
    @log_errors
    async def get_resume(self, resume_source: Any) -> str:
        """
        Parse a resume file and return its plain-text content.

        This method extracts text from resume chunks, handling both Document
        objects and plain strings. Empty or invalid chunks are skipped.

        Parameters
        ----------
        resume_source: Any
            Path, URL, or file-like object. Supports local paths, HTTP/HTTPS URLs,
            and HuggingFace Hub dataset references (e.g., "username/dataset::resume.pdf").

        Returns
        -------
        str
            Plain text content of the resume.

        Raises
        ------
        AssertionError
            If resume_source is None.
        Exception
            If parsing fails.
        """
        logger.info("Parsing resume...")
        resume_text = ""
        assert resume_source is not None, "resume_source cannot be None"

        if isinstance(resume_source, (str, Path)):
            resume_chunks = await get_resume_docs(resume_source)
        else:
            resume_chunks = self._parser(resume_source)

        for chunk in resume_chunks:
            if hasattr(chunk, "page_content") and chunk.page_content:
                resume_text += chunk.page_content
            elif isinstance(chunk, str) and chunk:
                resume_text += chunk
            else:
                logger.debug("Skipping empty or invalid chunk in resume: %s", chunk)

        return resume_text

    @log_async
    @log_errors
    async def _load_resume(self, resume_source: Any) -> str:
        """
        Load resume content, raising if the source is missing.

        This is a wrapper around get_resume() that validates the source first.
        Used by subgraph nodes for consistent error handling.

        Parameters
        ----------
        resume_source: Any
            Path or file-like object for the resume.

        Returns
        -------
        str
            Plain text content of the resume.

        Raises
        ------
        ValueError
            If resume_source is None or empty.
        """
        if not resume_source:
            raise ValueError("resume_source is required")
        return await self.get_resume(resume_source)

    async def _prompt_user_for_resume(self) -> str:
        """
        Prompt the user for input (synchronous input wrapped for async use).

        This method wraps the synchronous input() function to be used in async
        contexts. In a production async UI, this would be replaced with an
        async input mechanism.

        Note: This is a shared utility method. In a future refactoring, this
        could be extracted to a separate UserInputHelper class following the
        Interface Segregation Principle.

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
        return input("Please paste the resume in text format: ")
