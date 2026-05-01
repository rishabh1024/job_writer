"""AgentQL job-description scraper experiment.

Runs three extraction methods on 10 diverse job-posting URLs (30 trials):

* Method A — ``AQL_STRUCTURED``: bare AQL query, field names only (baseline)
* Method B — ``AQL_WITH_CONTEXT``: AQL query enriched with semantic context
  descriptions ``(...)`` per field and structural nesting for the main
  description section (AgentQL best-practice approach)
* Method C — ``PROMPT_EXPERIMENTAL``: free-form natural language prompt passed
  to ``get_data_by_prompt_experimental()``

Results are persisted to JSON and a comparison table is printed to the console.

Usage (PowerShell)::

    cd src\\job_writing_agent\\utils\\document_loader\\src
    python scraper_experiment.py

Output files are written to the ``experiment_results/`` directory alongside
this script:

* ``results_<timestamp>.json``   — full structured extraction data per URL
* ``experiment_<timestamp>.log`` — timestamped log of every scrape event
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

from job_writing_agent.utils.app_log.logging_config import (
    LoggingManager,
    get_logger,
)
from job_writing_agent.utils.app_log.logging_decorators import log_execution
from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    ExtractionMethod,
    JobExtract,
    ScraperError,
    extract_job_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR: Path = Path(__file__).parent / "experiment_results"

# 10 job postings across diverse job boards:
#   Greenhouse, Workday, Lever, Ashby, SmartRecruiters, direct company pages
JOB_URLS: list[str] = [
    # Greenhouse
    "https://job-boards.greenhouse.io/ocrolusinc/jobs/5837904004",
    # Lever
    "https://jobs.lever.co/openai/a1b2c3d4-0001-0001-0001-000000000001",
    # Workday
    (
        "https://amazon.jobs/en/jobs/2972591/"
        "software-development-engineer"
    ),
    # LinkedIn (public job page)
    "https://www.linkedin.com/jobs/view/software-engineer-at-google-3912345678",
    # Ashby
    "https://jobs.ashbyhq.com/anthropic/software-engineer",
    # SmartRecruiters
    (
        "https://jobs.smartrecruiters.com/Salesforce/"
        "software-engineer-backend"
    ),
    # Direct company career page — Microsoft
    (
        "https://careers.microsoft.com/us/en/job/1797500/"
        "Software-Engineer"
    ),
    # Direct company career page — Meta
    "https://www.metacareers.com/jobs/software-engineer-infrastructure",
    # iCIMS-hosted board
    (
        "https://careers-proofpoint.icims.com/jobs/5001/"
        "senior-software-engineer/job"
    ),
    # Greenhouse (second, different company)
    "https://job-boards.greenhouse.io/stripe/jobs/6309270",
]

# Number of tracked content fields in JobExtract (keep in sync with
# agentql_job_scraper._CONTENT_FIELDS)
_TOTAL_CONTENT_FIELDS: int = 12

# Console table column widths
_COL_URL: int = 55
_COL_METHOD: int = 22
_COL_FIELDS: int = 10
_COL_TIME: int = 10
_COL_STATUS: int = 8

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Captures the outcome of one URL × method trial.

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
    error_message: Optional[str] = None


@dataclass
class ExperimentReport:
    """Aggregated report of all trials in one experiment run.

    Attributes:
        run_id: ISO-8601 UTC timestamp identifying the run.
        total_trials: Total number of URL × method trials attempted.
        successful_trials: Count of trials that completed without error.
        results: All individual ``ExperimentResult`` instances.
    """

    run_id: str
    total_trials: int
    successful_trials: int
    results: list[ExperimentResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers — trial execution
# ---------------------------------------------------------------------------


def _run_single_trial(
    browser,
    url: str,
    method: ExtractionMethod,
) -> ExperimentResult:
    """Run one extraction trial and return an ``ExperimentResult``.

    Catches ``ScraperError`` and generic exceptions so the caller's loop
    continues even when individual trials fail.

    Args:
        browser: Launched Playwright ``Browser`` instance shared across trials.
        url: Job posting URL to scrape.
        method: Extraction method to apply.

    Returns:
        An ``ExperimentResult`` with ``is_success=True`` on success or
        ``is_success=False`` plus an ``error_message`` on failure.
    """
    try:
        extract = extract_job_data(browser, url, method)
        return ExperimentResult(
            url=url,
            method=method,
            extract=extract,
            is_success=True,
        )
    except ScraperError as exc:
        logger.error(
            "ScraperError for %s via %s: %s",
            url,
            method,
            exc.reason,
            exc_info=True,
        )
        failed_extract = JobExtract(
            url=url,
            method=method,
            has_error=True,
            error_message=str(exc),
        )
        return ExperimentResult(
            url=url,
            method=method,
            extract=failed_extract,
            is_success=False,
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error for %s via %s: %s",
            url,
            method,
            exc,
            exc_info=True,
        )
        failed_extract = JobExtract(
            url=url,
            method=method,
            has_error=True,
            error_message=str(exc),
        )
        return ExperimentResult(
            url=url,
            method=method,
            extract=failed_extract,
            is_success=False,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Private helpers — persistence
# ---------------------------------------------------------------------------


def _build_results_path(run_id: str) -> Path:
    """Construct the JSON output file path for a given run.

    Args:
        run_id: ISO-8601 UTC run identifier (colons replaced with hyphens for
            filesystem compatibility).

    Returns:
        Absolute ``Path`` to the JSON results file.
    """
    safe_id = run_id.replace(":", "-").replace(" ", "_")
    return RESULTS_DIR / f"results_{safe_id}.json"


def _build_log_path(run_id: str) -> Path:
    """Construct the log file path for a given run.

    Args:
        run_id: ISO-8601 UTC run identifier.

    Returns:
        Absolute ``Path`` to the ``.log`` file.
    """
    safe_id = run_id.replace(":", "-").replace(" ", "_")
    return RESULTS_DIR / f"experiment_{safe_id}.log"


def _serialize_report(report: ExperimentReport) -> dict:
    """Serialise an ``ExperimentReport`` to a plain dict for JSON output.

    ``JobExtract`` dataclasses are converted via ``dataclasses.asdict``.
    The private ``_CONTENT_FIELDS`` tuple is excluded from the output.

    Args:
        report: The completed experiment report.

    Returns:
        A JSON-serialisable ``dict``.
    """
    results_list = []
    for result in report.results:
        extract_dict = asdict(result.extract)
        extract_dict.pop("_CONTENT_FIELDS", None)
        results_list.append(
            {
                "url": result.url,
                "method": result.method,
                "is_success": result.is_success,
                "error_message": result.error_message,
                "extract": extract_dict,
            }
        )
    return {
        "run_id": report.run_id,
        "total_trials": report.total_trials,
        "successful_trials": report.successful_trials,
        "results": results_list,
    }


@log_execution
def save_report(report: ExperimentReport, output_path: Path) -> None:
    """Write the experiment report to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        report: Completed ``ExperimentReport``.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_report(report)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Private helpers — console display
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Return ``text`` truncated to ``max_len`` chars with ellipsis if needed.

    Args:
        text: Input string.
        max_len: Maximum allowed length.

    Returns:
        Original string or a truncated version ending in ``…``.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _print_summary_table(report: ExperimentReport) -> None:
    """Print a formatted comparison table of all results to stdout.

    Args:
        report: Completed ``ExperimentReport`` to display.
    """
    sep = (
        "-" * _COL_URL
        + "-+-"
        + "-" * _COL_METHOD
        + "-+-"
        + "-" * _COL_FIELDS
        + "-+-"
        + "-" * _COL_TIME
        + "-+-"
        + "-" * _COL_STATUS
    )
    header = (
        f"{'URL':<{_COL_URL}} | "
        f"{'Method':<{_COL_METHOD}} | "
        f"{'Fields':<{_COL_FIELDS}} | "
        f"{'Time(ms)':<{_COL_TIME}} | "
        f"{'Status':<{_COL_STATUS}}"
    )

    print()
    print(
        f"Experiment run: {report.run_id}  |  "
        f"Trials: {report.total_trials}  |  "
        f"Successful: {report.successful_trials}"
    )
    print(sep)
    print(header)
    print(sep)

    for result in report.results:
        status = "OK" if result.is_success else "FAIL"
        fields_label = (
            f"{result.extract.populated_fields}/{_TOTAL_CONTENT_FIELDS}"
            if result.is_success
            else "—"
        )
        time_label = (
            str(result.extract.scrape_time_ms)
            if result.is_success
            else "—"
        )
        row = (
            f"{_truncate(result.url, _COL_URL):<{_COL_URL}} | "
            f"{result.method:<{_COL_METHOD}} | "
            f"{fields_label:<{_COL_FIELDS}} | "
            f"{time_label:<{_COL_TIME}} | "
            f"{status:<{_COL_STATUS}}"
        )
        print(row)

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@log_execution
def run_experiment(job_urls: list[str]) -> ExperimentReport:
    """Run structured and natural-language extractions for every URL.

    Launches a single headless Chromium instance shared across all trials to
    avoid the overhead of repeated browser starts.  Each trial opens and
    closes its own page tab.

    Args:
        job_urls: List of public job-posting URLs to scrape.

    Returns:
        A completed ``ExperimentReport`` with results for every trial.
    """
    run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    methods = [
        ExtractionMethod.AQL_STRUCTURED,
        ExtractionMethod.AQL_WITH_CONTEXT,
        ExtractionMethod.PROMPT_EXPERIMENTAL,
    ]
    total_trials = len(job_urls) * len(methods)
    results: list[ExperimentResult] = []

    logger.info(
        "Experiment started: run_id=%s urls=%d methods=%d trials=%d",
        run_id,
        len(job_urls),
        len(methods),
        total_trials,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for url in job_urls:
                for method in methods:
                    logger.info(
                        "Running trial: url=%s method=%s", url, method
                    )
                    result = _run_single_trial(browser, url, method)
                    results.append(result)
        finally:
            browser.close()

    successful = sum(1 for r in results if r.is_success)
    report = ExperimentReport(
        run_id=run_id,
        total_trials=total_trials,
        successful_trials=successful,
        results=results,
    )
    logger.info(
        "Experiment finished: run_id=%s successful=%d/%d",
        run_id,
        successful,
        total_trials,
    )
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure logging, run the experiment, persist results, and print table.

    Logging is initialised here (once) with both a console handler and a
    per-run log file in ``experiment_results/``.
    """
    run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    log_path = _build_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    LoggingManager().configure_logging(
        log_level=logging.DEBUG,
        log_file=log_path,
    )

    logger.info("=== AgentQL Job Scraper Experiment ===")

    report = run_experiment(JOB_URLS)

    results_path = _build_results_path(report.run_id)
    save_report(report, results_path)

    _print_summary_table(report)

    print(f"Full results : {results_path}")
    print(f"Log file     : {log_path}")


if __name__ == "__main__":
    main()
