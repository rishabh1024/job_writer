"""Context-enriched AQL strategy — semantic + structural hints on every field.

This is the Method B recommended approach, applying both AgentQL best
practices from https://docs.agentql.com/agentql-query/best-practices:

1. Semantic context — a ``(description)`` on every field term helps AgentQL
   disambiguate when multiple candidates match the same name.
2. Structural context — list fields are nested under
   ``job_description_section`` to prevent picking up unrelated lists such as
   nav menus or footer columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from job_writing_agent.utils.document_loader.src._constants import (
    JOB_DESCRIPTION_QUERY_WITH_CONTEXT,
    ExtractionMethod,
)
from job_writing_agent.utils.document_loader.src.strategies.base import (
    BaseScraperStrategy,
)

if TYPE_CHECKING:
    from agentql.ext.playwright.sync_api import Page


class AqlWithContextStrategy(BaseScraperStrategy):
    """Extract job data using a context-enriched AQL query.

    Calls ``page.query_data()`` with ``JOB_DESCRIPTION_QUERY_WITH_CONTEXT``,
    which annotates every field with a parenthesised semantic description and
    nests list fields inside a ``job_description_section`` container.

    The raw response has a nested shape:
    ``{"job_description_section": {"responsibilities": [...], ...}, ...}``

    ``AgentQlJobScraper`` passes this response through
    ``_flatten_context_response()`` before building a ``JobExtract``.
    """

    @property
    def method_name(self) -> ExtractionMethod:
        """Return the ``AQL_WITH_CONTEXT`` method identifier."""
        return ExtractionMethod.AQL_WITH_CONTEXT

    @property
    def description(self) -> str:
        """Return a short description of this strategy."""
        return (
            "AQL query with semantic context hints "
            "and structural nesting (best practice)"
        )

    def execute(self, page: Page) -> dict:
        """Run the context-enriched AQL query against the current page.

        Args:
            page: AgentQL-wrapped Playwright ``Page`` at the target URL.

        Returns:
            Nested ``dict`` with top-level metadata fields and a
            ``job_description_section`` sub-dict containing list fields.
        """
        return page.query_data(JOB_DESCRIPTION_QUERY_WITH_CONTEXT)
