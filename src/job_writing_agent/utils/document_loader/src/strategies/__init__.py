"""Scraper strategy implementations.

Public exports:

- ``BaseScraperStrategy``    -- abstract base class all strategies must
  implement
- ``AqlStructuredStrategy``  -- bare AQL query, no context hints (baseline)
- ``AqlWithContextStrategy`` -- AQL query with semantic + structural context
- ``PromptExperimentalStrategy`` -- free-form NL prompt

Usage::

    from job_writing_agent.utils.document_loader.src.strategies import (
        AqlWithContextStrategy,
    )
    strategy = AqlWithContextStrategy()
    raw = strategy.execute(page)
"""

from __future__ import annotations

from job_writing_agent.utils.document_loader.src.strategies.aql_structured import (  # noqa: E501
    AqlStructuredStrategy,
)
from job_writing_agent.utils.document_loader.src.strategies.aql_with_context import (  # noqa: E501
    AqlWithContextStrategy,
)
from job_writing_agent.utils.document_loader.src.strategies.base import (
    BaseScraperStrategy,
)
from job_writing_agent.utils.document_loader.src.strategies.prompt_experimental import (  # noqa: E501
    PromptExperimentalStrategy,
)

__all__ = [
    "AqlStructuredStrategy",
    "AqlWithContextStrategy",
    "BaseScraperStrategy",
    "PromptExperimentalStrategy",
]
