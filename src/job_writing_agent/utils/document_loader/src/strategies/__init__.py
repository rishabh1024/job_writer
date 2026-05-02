"""Scraper strategy implementations.

Public exports:

- ``BaseScraperStrategy``    -- abstract base class all strategies must
  implement
- ``AqlWithContextStrategy`` -- AQL query with semantic + structural context

Usage::

    from job_writing_agent.utils.document_loader.src.strategies import (
        AqlWithContextStrategy,
    )
    strategy = AqlWithContextStrategy()
    raw = strategy.execute(page)
"""

from __future__ import annotations

from job_writing_agent.utils.document_loader.src.strategies.aql_with_context import (  # noqa: E501
    AqlWithContextStrategy,
)
from job_writing_agent.utils.document_loader.src.strategies.base import (
    BaseScraperStrategy,
)

__all__ = [
    "AqlWithContextStrategy",
    "BaseScraperStrategy",
]
