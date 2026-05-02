"""Shared data models for the scraper experiment.

These dataclasses are intentionally kept in a dedicated module so that both
``scraper_experiment.py`` (runner) and ``results/report_builder.py`` (output)
can import them without creating a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (  # noqa: E501
        ExtractionMethod,
        JobExtract,
    )


@dataclass
class ExperimentResult:
    """Captures the outcome of one URL x strategy trial.

    Attributes:
        url: Source URL of the job posting.
        method: Extraction method used.
        extract: The ``JobExtract`` produced (always present, even on error).
        is_success: ``True`` when extraction completed without exception.
        error_message: Exception message when ``is_success`` is ``False``.
    """

    url: str
    method: ExtractionMethod
    extract: JobExtract
    is_success: bool
    error_message: str | None = None


@dataclass
class ExperimentReport:
    """Aggregated report of all trials in one experiment run.

    Attributes:
        run_id: ISO-8601 UTC timestamp identifying the run.
        total_trials: Total number of URL x strategy trials attempted.
        successful_trials: Count of trials that completed without error.
        results: All individual ``ExperimentResult`` instances, ordered by
            URL then strategy (matching the execution order).
    """

    run_id: str
    total_trials: int
    successful_trials: int
    results: list[ExperimentResult] = field(default_factory=list)
