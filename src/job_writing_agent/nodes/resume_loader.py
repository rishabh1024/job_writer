# -*- coding: utf-8 -*-
"""
Resume Loader Module

This module provides the ResumeLoader class responsible for loading and parsing
the resume file and returning the resume in the required format.
"""

import logging
from typing import Any, Callable, Optional

from typing_extensions import deprecated

from job_writing_agent.utils.app_log.logging_decorators import log_errors
from job_writing_agent.utils.document_processing import parse_resume

logger = logging.getLogger(__name__)


class ResumeLoader:
    """
    Responsible for loading and parsing resume documents.

    Example:
        >>> loader = ResumeLoader()
        >>> resume_text = await loader.load_resume("path/to/resume.pdf")
        >>>
        >>> # With custom parser for testing
        >>> mock_parser = lambda x: "mock resume text"
        >>> loader = ResumeLoader(parser=mock_parser)
    """

    def __init__(self, parser: Optional[Callable[[Any], Any]] = None):
        """
        Initialize ResumeLoader with optional parser dependency injection.

        Parameters
        ----------
        parser: Optional[Callable[[Any], Any]]
            Function to parse resume and return str. Defaults to `parse_resume`
            from document_processing. Can be injected for testing.
        """
        self._parser = parser or parse_resume

    @log_errors
    async def load_resume(self, resume_source: Any) -> str:
        """
        Load a resume from the given source and return its plain-text content.

        Parameters
        ----------
        resume_source: Any
            Path to a .pdf or .txt resume file (or file-like). Callers should
            pass a non-empty source; validation is typically done in the node.

        Returns
        -------
        str
            Plain text content of the resume.

        Raises
        ------
        Exception
            If loading/parsing fails.
        """
        logger.info("Loading resume...")
        result = self._parser(resume_source)
        return result if isinstance(result, str) else ""

    @deprecated(
        "Resume prompting now uses LangGraph interrupts (prompt_user_for_resume_node).",
        category=DeprecationWarning,
    )
    async def _prompt_user_for_resume(self) -> str:
        """
        Prompt the user for resume text via stdin.

        Kept for backward compatibility only. Use prompt_user_for_resume_node with
        LangGraph interrupt in new code.

        Returns
        -------
        str
            User input string.
        """
        return input("Please paste the resume in text format: ")
