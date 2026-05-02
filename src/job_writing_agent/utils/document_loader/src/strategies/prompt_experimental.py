"""Experimental natural-language prompt strategy.

Uses ``page.get_data_by_prompt_experimental()``, a separate AgentQL inference
path that accepts free-form English rather than an AQL query.  This method C
tests whether an unstructured prompt can match or exceed the accuracy of a
carefully authored AQL query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from job_writing_agent.utils.document_loader.src._constants import (
    JOB_DESCRIPTION_PROMPT,
    ExtractionMethod,
)
from job_writing_agent.utils.document_loader.src.strategies.base import (
    BaseScraperStrategy,
)

if TYPE_CHECKING:
    from agentql.ext.playwright.sync_api import Page


class PromptExperimentalStrategy(BaseScraperStrategy):
    """Extract job data using a free-form natural-language prompt.

    Calls ``page.get_data_by_prompt_experimental()`` with
    ``JOB_DESCRIPTION_PROMPT``.  Unlike the AQL strategies this method does
    not require a structured query; the AgentQL backend interprets the prompt
    directly and returns a best-effort ``dict``.

    Note:
        The returned ``dict`` has an unspecified shape — key names are
        inferred by AgentQL and may differ from the canonical field names used
        by the AQL strategies.  ``AgentQlJobScraper._parse_aql_response``
        maps known keys via ``flat.get(field_name)`` so unrecognised keys are
        silently ignored.
    """

    @property
    def method_name(self) -> ExtractionMethod:
        """Return the ``PROMPT_EXPERIMENTAL`` method identifier."""
        return ExtractionMethod.PROMPT_EXPERIMENTAL

    @property
    def description(self) -> str:
        """Return a short description of this strategy."""
        return "Free-form NL prompt via get_data_by_prompt_experimental()"

    def execute(self, page: Page) -> dict:
        """Run the experimental prompt against the current page.

        Args:
            page: AgentQL-wrapped Playwright ``Page`` at the target URL.

        Returns:
            Best-effort ``dict`` whose keys are inferred by AgentQL from the
            natural-language prompt.
        """
        return page.get_data_by_prompt_experimental(JOB_DESCRIPTION_PROMPT)
