"""Experiment result formatters.

Public exports:

- ``build_markdown_report`` — render an ``ExperimentReport`` as a markdown string
- ``save_markdown_report`` — write the markdown report to a ``.md`` file
"""

from job_writing_agent.utils.document_loader.src.results.report_builder import (
    build_markdown_report,
    save_markdown_report,
)

__all__ = ["build_markdown_report", "save_markdown_report"]
