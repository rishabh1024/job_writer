"""Abstract base class for all AgentQL job-description scraper strategies.

Every concrete strategy encapsulates exactly one extraction approach: the
query or prompt constant it owns, and the single AgentQL API call needed to
produce a raw ``dict``.  The caller (``AgentQlJobScraper``) is completely
decoupled from which strategy is in use — it only calls ``execute()``.

Adding a new strategy requires only a new file that subclasses
``BaseScraperStrategy``; no changes to the scraper or experiment runner are
necessary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentql.ext.playwright.sync_api import Page

from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    ExtractionMethod,
)


class BaseScraperStrategy(ABC):
    """Contract that every extraction strategy must fulfil.

    Subclasses own their query/prompt constant and the single AgentQL call.
    They must not manage page navigation or lifecycle — that responsibility
    belongs to ``AgentQlJobScraper``.

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
            Raw ``dict`` returned by AgentQL.  Shape varies by strategy:
            flat for ``AQL_STRUCTURED`` / ``PROMPT_EXPERIMENTAL``, nested
            under ``job_description_section`` for ``AQL_WITH_CONTEXT``.

        Raises:
            Exception: Any exception from the AgentQL SDK is propagated as-is
                so ``AgentQlJobScraper`` can wrap it in a ``ScraperError``.
        """
