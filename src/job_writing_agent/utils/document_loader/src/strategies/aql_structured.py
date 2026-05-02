"""Bare AQL structured strategy — field names only, no context hints.

This is the Method A baseline.  Results can be directly compared against
``AqlWithContextStrategy`` to measure the accuracy uplift from adding
semantic and structural context to the query.
"""

from __future__ import annotations

from agentql.ext.playwright.sync_api import Page

from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    ExtractionMethod,
    JOB_DESCRIPTION_QUERY,
)
from job_writing_agent.utils.document_loader.src.strategies.base import (
    BaseScraperStrategy,
)


class AqlStructuredStrategy(BaseScraperStrategy):
    """Extract job data using a bare AQL query with no context hints.

    Calls ``page.query_data()`` with ``JOB_DESCRIPTION_QUERY``, which
    contains field names only.  AgentQL must infer element locations purely
    from the term names.
    """

    @property
    def method_name(self) -> ExtractionMethod:
        """Return the ``AQL_STRUCTURED`` method identifier."""
        return ExtractionMethod.AQL_STRUCTURED

    @property
    def description(self) -> str:
        """Return a short description of this strategy."""
        return "Bare AQL query — field names only (baseline)"

    def execute(self, page: Page) -> dict:
        """Run the bare AQL query against the current page.

        Args:
            page: AgentQL-wrapped Playwright ``Page`` at the target URL.

        Returns:
            Flat ``dict`` keyed by the field names in ``JOB_DESCRIPTION_QUERY``.
        """
        return page.query_data(JOB_DESCRIPTION_QUERY)
