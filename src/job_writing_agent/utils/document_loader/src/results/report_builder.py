"""Markdown report builder for the AgentQL scraper experiment.

Generates a human-readable ``.md`` file structured as:

    # Experiment Report — <run_id>

    ## Summary

    | Strategy | Trials | Success | Error | Avg Fields | Avg Time (ms) |
    |...|

    ---

    ## URL 1 — <url>

    ### Strategy: AQL Structured (baseline)
    - **Status**: OK
    - **Fields extracted**: 9 / 12
    - **Scrape time**: 4 200 ms
    | Field | Value |
    |---|---|
    | job_title | Staff Engineer |
    ...

    ### Strategy: AQL With Context (best practice)
    ...

    ### Strategy: Prompt Experimental
    ...

    ---

    ## URL 2 — <url>
    ...

This layout renders cleanly on GitHub, in VS Code preview, and in any markdown
viewer.  Long text fields (job_summary, responsibilities, etc.) are preserved
in full — no truncation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from job_writing_agent.utils.app_log.logging_config import get_logger
from job_writing_agent.utils.app_log.logging_decorators import log_execution
from job_writing_agent.utils.document_loader.src.strategies import (
    AqlStructuredStrategy,
    AqlWithContextStrategy,
    PromptExperimentalStrategy,
)

if TYPE_CHECKING:
    from pathlib import Path

    from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (  # noqa: E501
        JobExtract,
    )
    from job_writing_agent.utils.document_loader.src.experiment_models import (
        ExperimentReport,
        ExperimentResult,
    )

logger = get_logger(__name__)

_FIELD_LABELS: list[tuple[str, str]] = [
    ("job_title", "Job Title"),
    ("company_name", "Company"),
    ("job_location", "Location"),
    ("employment_type", "Employment Type"),
    ("salary_range", "Salary Range"),
    ("remote_policy", "Remote Policy"),
    ("application_deadline", "Application Deadline"),
    ("job_summary", "Job Summary"),
    ("responsibilities", "Responsibilities"),
    ("requirements", "Requirements"),
    ("preferred_qualifications", "Preferred Qualifications"),
    ("benefits", "Benefits"),
]

_TOTAL_FIELDS: int = len(_FIELD_LABELS)


def _render_field_value(value: object) -> str:
    """Convert a field value to a markdown-safe display string.

    Lists are rendered as a markdown bullet list (one item per line).
    ``None`` and empty collections are rendered as an italic ``_not found_``.

    Args:
        value: The raw field value from a ``JobExtract``.

    Returns:
        A markdown string ready for embedding in a table cell or paragraph.
    """
    if not value:
        return "_not found_"
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            return "_not found_"
        return "<br>".join(f"• {item}" for item in items)
    return str(value).strip()


def _render_field_table(extract: JobExtract) -> str:
    """Build a two-column markdown table of all field values.

    Args:
        extract: The ``JobExtract`` whose fields to render.

    Returns:
        Multi-line markdown string containing the ``| Field | Value |`` table.
    """
    rows = ["| Field | Value |", "|---|---|"]
    for attr, label in _FIELD_LABELS:
        value = getattr(extract, attr)
        rows.append(f"| **{label}** | {_render_field_value(value)} |")
    return "\n".join(rows)


def _render_strategy_section(result: ExperimentResult) -> str:
    """Render one ``### Strategy: ...`` section for a single trial.

    Args:
        result: The ``ExperimentResult`` for this URL x strategy trial.

    Returns:
        Multi-line markdown string for this strategy section.
    """
    strategy_descriptions = {
        "aql_structured": AqlStructuredStrategy().description,
        "aql_with_context": AqlWithContextStrategy().description,
        "prompt_experimental": PromptExperimentalStrategy().description,
    }

    method_label = str(result.method).replace("_", " ").title()
    description = strategy_descriptions.get(
        str(result.method), str(result.method),
    )

    lines: list[str] = [
        f"### Strategy: {method_label}",
        "",
        f"> {description}",
        "",
    ]

    if not result.is_success:
        lines += [
            "**Status**: FAILED",
            "",
            f"**Error**: `{result.error_message}`",
            "",
        ]
        return "\n".join(lines)

    extract = result.extract
    completeness_pct = int(
        (extract.populated_fields / _TOTAL_FIELDS) * 100,
    )
    lines += [
        f"**Status**: OK  |  "
        f"**Fields extracted**: {extract.populated_fields} / {_TOTAL_FIELDS}"
        f" ({completeness_pct}%)  |  "
        f"**Scrape time**: {extract.scrape_time_ms:,} ms",
        "",
        _render_field_table(extract),
        "",
    ]
    return "\n".join(lines)


def _render_url_section(url: str, results: list[ExperimentResult]) -> str:
    """Render one ``## URL N — <url>`` section containing all strategy blocks.

    Args:
        url: The job-posting URL for this group.
        results: All ``ExperimentResult`` instances for this URL, one per
            strategy, in execution order.

    Returns:
        Multi-line markdown string for this URL section.
    """
    lines: list[str] = [
        f"## {url}",
        "",
    ]
    for result in results:
        lines.append(_render_strategy_section(result))
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _render_summary_table(report: ExperimentReport) -> str:
    """Build the aggregate summary table across all strategies.

    Groups results by strategy and computes: trial count, success count,
    error count, average field completeness (%), and average scrape time (ms).

    Args:
        report: The completed ``ExperimentReport``.

    Returns:
        Multi-line markdown string for the summary table.
    """
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "trials": 0,
            "success": 0,
            "error": 0,
            "total_fields": 0,
            "total_time_ms": 0,
        },
    )

    for result in report.results:
        method_key = str(result.method)
        stats[method_key]["trials"] += 1
        if result.is_success:
            stats[method_key]["success"] += 1
            stats[method_key]["total_fields"] += result.extract.populated_fields
            stats[method_key]["total_time_ms"] += result.extract.scrape_time_ms
        else:
            stats[method_key]["error"] += 1

    rows = [
        "| Strategy | Trials | Success | Errors |"
        " Avg Completeness | Avg Time (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for method_key, method_stats in stats.items():
        label = method_key.replace("_", " ").title()
        success_count = method_stats["success"]
        avg_completeness = (
            f"{method_stats['total_fields'] / success_count / _TOTAL_FIELDS * 100:.0f}%"  # noqa: E501
            if success_count
            else "—"
        )
        avg_time_ms = (
            f"{method_stats['total_time_ms'] // success_count:,}"
            if success_count
            else "—"
        )
        rows.append(
            f"| {label} | {method_stats['trials']} | {success_count}"
            f" | {method_stats['error']} |"
            f" {avg_completeness} | {avg_time_ms} |",
        )
    return "\n".join(rows)


@log_execution
def build_markdown_report(report: ExperimentReport) -> str:
    """Render a complete markdown experiment report as a string.

    The report structure is:

    * Title + run metadata
    * Summary aggregate table (one row per strategy)
    * One ``## URL N`` section per URL, each containing one
      ``### Strategy: ...`` block per strategy

    Args:
        report: The completed ``ExperimentReport`` to render.

    Returns:
        A multi-line markdown string ready to be written to a ``.md`` file.
    """
    lines: list[str] = [
        "# AgentQL Job Description Scraper — Experiment Report",
        "",
        f"**Run ID**: `{report.run_id}`  |  "
        f"**Total trials**: {report.total_trials}  |  "
        f"**Successful**: {report.successful_trials} / {report.total_trials}",
        "",
        "## Summary",
        "",
        _render_summary_table(report),
        "",
        "---",
        "",
    ]

    url_to_results: dict[str, list[ExperimentResult]] = {}
    for result in report.results:
        url_to_results.setdefault(result.url, []).append(result)

    for url, results in url_to_results.items():
        lines.append(_render_url_section(url, results))

    return "\n".join(lines)


@log_execution
def save_markdown_report(report: ExperimentReport, output_path: Path) -> None:
    """Write the markdown experiment report to a ``.md`` file.

    Creates parent directories if they do not exist.

    Args:
        report: Completed ``ExperimentReport``.
        output_path: Destination file path (should end in ``.md``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_markdown_report(report)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Markdown report saved to %s", output_path)
