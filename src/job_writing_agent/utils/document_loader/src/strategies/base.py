"""Abstract base class for AgentQL job-description scraper strategies.

The active strategy encapsulates one extraction approach: the query constant
it owns and the single AgentQL API call needed to produce a raw ``dict``. The
caller (``AgentQlJobScraper``) is decoupled from those details and only calls
``execute()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentql.ext.playwright.sync_api import Page

    from job_writing_agent.utils.document_loader.src._constants import (
        ExtractionMethod,
    )


class BaseScraperStrategy(ABC):
    """Contract that an extraction strategy must fulfil.

    Subclasses own their query constant and the single AgentQL call. They must
    not manage page navigation or lifecycle — that responsibility belongs to
    ``AgentQlJobScraper``.

    Example:
        >>> strategy = AqlWithContextStrategy()
        >>> raw = strategy.execute(page)
        >>> print(raw["job_title"])
    """

    @property
    @abstractmethod
    def method_name(self) -> ExtractionMethod:
        """The ``ExtractionMethod`` enum value that identifies this strategy.

        Returns:
            The corresponding ``ExtractionMethod`` member.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable one-line description of what this strategy does.

        Returns:
            A short string suitable for logging and report headers.
        """

    @abstractmethod
    def execute(self, page: Page) -> dict:
        """Run the extraction against an already-navigated page.

        The page must already be at the target URL before this method is
        called.  The method must not navigate, reload, or close the page.

        Args:
            page: AgentQL-wrapped Playwright ``Page`` at the target URL.

        Returns:
            Raw ``dict`` returned by AgentQL, usually with a top-level
            ``body[]`` wrapper for the AQL response.

        Raises:
            Exception: Any exception from the AgentQL SDK is propagated as-is
                so ``AgentQlJobScraper`` can wrap it in a ``ScraperError``.
        """
