r"""AgentQL job-description scraper experiment.

Runs three extraction strategies on 10 diverse job-posting URLs (30 trials):

* Strategy A — ``AqlStructuredStrategy``: bare AQL query, field names only
* Strategy B — ``AqlWithContextStrategy``: AQL query enriched with semantic
  context and structural nesting (AgentQL best-practice approach)
* Strategy C — ``PromptExperimentalStrategy``: free-form NL prompt via
  ``get_data_by_prompt_experimental()``

Usage (PowerShell)::

    cd src\\job_writing_agent\\utils\\document_loader\\src
    python scraper_experiment.py

Output files are written to the ``experiment_results/`` directory:

* ``results_<timestamp>.json``    — full structured extraction data
* ``report_<timestamp>.md``       — human-readable per-URL per-strategy report
* ``experiment_<timestamp>.log``  — timestamped log of every scrape event
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import Browser, sync_playwright

from job_writing_agent.utils.app_log.logging_config import (
    LoggingManager,
    get_logger,
)
from job_writing_agent.utils.app_log.logging_decorators import log_execution
from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    AgentQlJobScraper,
    JobExtract,
    ScraperError,
)
from job_writing_agent.utils.document_loader.src.experiment_models import (
    ExperimentReport,
    ExperimentResult,
)
from job_writing_agent.utils.document_loader.src.results import (
    save_markdown_report,
)
from job_writing_agent.utils.document_loader.src.strategies import (
    AqlStructuredStrategy,
    AqlWithContextStrategy,
    BaseScraperStrategy,
    PromptExperimentalStrategy,
)

RESULTS_DIR: Path = Path(__file__).parent / "experiment_results"

JOB_URLS: list[str] = [
    # Greenhouse
    "https://job-boards.greenhouse.io/ocrolusinc/jobs/5837904004",
    # Lever
    "https://jobs.lever.co/openai/a1b2c3d4-0001-0001-0001-000000000001",
    # Workday / Amazon Jobs
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
    # Microsoft careers
    (
        "https://careers.microsoft.com/us/en/job/1797500/"
        "Software-Engineer"
    ),
    # Meta careers
    "https://www.metacareers.com/jobs/software-engineer-infrastructure",
    # iCIMS-hosted board
    (
        "https://careers-proofpoint.icims.com/jobs/5001/"
        "senior-software-engineer/job"
    ),
    # Greenhouse (second company)
    "https://job-boards.greenhouse.io/stripe/jobs/6309270",
]

_TOTAL_CONTENT_FIELDS: int = 12

_COL_URL: int = 55
_COL_STRATEGY: int = 24
_COL_FIELDS: int = 10
_COL_TIME: int = 10
_COL_STATUS: int = 8

logger = get_logger(__name__)


def _run_single_trial(
    browser: Browser,
    url: str,
    strategy: BaseScraperStrategy,
) -> ExperimentResult:
    """Run one extraction trial and return an ``ExperimentResult``.

    Catches ``ScraperError`` and generic exceptions so the caller's loop
    continues even when individual trials fail.

    Args:
        browser: Launched Playwright ``Browser`` instance shared across trials.
        url: Job posting URL to scrape.
        strategy: Concrete ``BaseScraperStrategy`` instance to apply.

    Returns:
        An ``ExperimentResult`` with ``is_success=True`` on success or
        ``is_success=False`` plus an ``error_message`` on failure.
    """
    method = strategy.method_name
    scraper = AgentQlJobScraper(browser, strategy)
    try:
        extract = scraper.scrape(url)
        return ExperimentResult(
            url=url,
            method=method,
            extract=extract,
            is_success=True,
        )
    except ScraperError as exc:
        logger.exception(
            "ScraperError for %s via %s: %s",
            url,
            method,
            exc.reason,
        )
        return ExperimentResult(
            url=url,
            method=method,
            extract=JobExtract(
                url=url,
                method=method,
                has_error=True,
                error_message=str(exc),
            ),
            is_success=False,
            error_message=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error for %s via %s", url, method)
        return ExperimentResult(
            url=url,
            method=method,
            extract=JobExtract(
                url=url,
                method=method,
                has_error=True,
                error_message=str(exc),
            ),
            is_success=False,
            error_message=str(exc),
        )


def _build_json_path(run_id: str) -> Path:
    """Construct the JSON output file path for a given run.

    Args:
        run_id: ISO-8601 UTC run identifier.

    Returns:
        Absolute ``Path`` to the JSON results file.
    """
    safe_id = run_id.replace(":", "-").replace(" ", "_")
    return RESULTS_DIR / f"results_{safe_id}.json"


def _build_markdown_path(run_id: str) -> Path:
    """Construct the markdown report file path for a given run.

    Args:
        run_id: ISO-8601 UTC run identifier.

    Returns:
        Absolute ``Path`` to the ``.md`` report file.
    """
    safe_id = run_id.replace(":", "-").replace(" ", "_")
    return RESULTS_DIR / f"report_{safe_id}.md"


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

    Args:
        report: The completed experiment report.

    Returns:
        A JSON-serialisable ``dict``.
    """
    serialized_results = []
    for result in report.results:
        extract_dict = asdict(result.extract)
        extract_dict.pop("_CONTENT_FIELDS", None)
        serialized_results.append(
            {
                "url": result.url,
                "method": result.method,
                "is_success": result.is_success,
                "error_message": result.error_message,
                "extract": extract_dict,
            },
        )
    return {
        "run_id": report.run_id,
        "total_trials": report.total_trials,
        "successful_trials": report.successful_trials,
        "results": serialized_results,
    }


@log_execution
def save_json_report(report: ExperimentReport, output_path: Path) -> None:
    """Write the experiment report to a JSON file.

    Args:
        report: Completed ``ExperimentReport``.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_report(report)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, ensure_ascii=False)
    logger.info("JSON results saved to %s", output_path)


def _truncate(text: str, max_len: int) -> str:
    """Return ``text`` truncated to ``max_len`` chars with an ellipsis.

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
        + "-" * _COL_STRATEGY
        + "-+-"
        + "-" * _COL_FIELDS
        + "-+-"
        + "-" * _COL_TIME
        + "-+-"
        + "-" * _COL_STATUS
    )
    header = (
        f"{'URL':<{_COL_URL}} | "
        f"{'Strategy':<{_COL_STRATEGY}} | "
        f"{'Fields':<{_COL_FIELDS}} | "
        f"{'Time(ms)':<{_COL_TIME}} | "
        f"{'Status':<{_COL_STATUS}}"
    )

    print()
    print(
        f"Experiment run : {report.run_id}  |  "
        f"Trials: {report.total_trials}  |  "
        f"Successful: {report.successful_trials}",
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
            f"{result.method!s:<{_COL_STRATEGY}} | "
            f"{fields_label:<{_COL_FIELDS}} | "
            f"{time_label:<{_COL_TIME}} | "
            f"{status:<{_COL_STATUS}}"
        )
        print(row)

    print(sep)
    print()


@log_execution
def run_experiment(job_urls: list[str]) -> ExperimentReport:
    """Run all three strategies against every URL.

    Launches a single headless Chromium instance shared across all trials.
    Each trial opens and closes its own page tab.

    Args:
        job_urls: List of public job-posting URLs to scrape.

    Returns:
        A completed ``ExperimentReport`` with results for every trial.
    """
    run_id = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    strategies: list[BaseScraperStrategy] = [
        AqlStructuredStrategy(),
        AqlWithContextStrategy(),
        PromptExperimentalStrategy(),
    ]
    total_trials = len(job_urls) * len(strategies)
    results: list[ExperimentResult] = []

    logger.info(
        "Experiment started: run_id=%s urls=%d strategies=%d trials=%d",
        run_id,
        len(job_urls),
        len(strategies),
        total_trials,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for url in job_urls:
                for strategy in strategies:
                    logger.info(
                        "Running trial: url=%s strategy=%s",
                        url,
                        strategy.method_name,
                    )
                    result = _run_single_trial(browser, url, strategy)
                    results.append(result)
        finally:
            browser.close()

    successful = sum(1 for trial_result in results if trial_result.is_success)
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


def main() -> None:
    """Configure logging, run the experiment, persist results, print table."""
    run_id = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    log_path = _build_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    LoggingManager().configure_logging(
        log_level=logging.DEBUG,
        log_file=log_path,
    )

    logger.info("=== AgentQL Job Scraper Experiment ===")

    report = run_experiment(JOB_URLS)

    json_path = _build_json_path(report.run_id)
    save_json_report(report, json_path)

    md_path = _build_markdown_path(report.run_id)
    save_markdown_report(report, md_path)

    _print_summary_table(report)

    print(f"JSON results : {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Log file     : {log_path}")


if __name__ == "__main__":
    main()
