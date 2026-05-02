"""AgentQL job-description scraper experiment.

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
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR: Path = Path(__file__).parent / "experiment_results"

# 10 job postings across diverse job boards:
#   Greenhouse, Workday, Lever, Ashby, SmartRecruiters, direct company pages
JOB_URLS: list[str] = [
    "https://autodesk.wd1.myworkdayjobs.com/en-US/Ext/job/Pune%2C-IND/Senior-Software-Engineer_25WD93636-1?src=JB-10065&source=LinkedIn",
    "https://paypal.wd1.myworkdayjobs.com/en-US/jobs/job/Bangalore-Karnataka-India/Senior-Software-Engineer---Backend--Java-_R0134858",
    "https://fox.wd1.myworkdayjobs.com/en-US/Domestic/job/IND-KA-Bengaluru/Senior-Software-Development-Engineer--Backend_R50031537",
    "https://altera.wd1.myworkdayjobs.com/en-US/altera/job/Bengaluru-Karnataka-India/FPGA-IP-Software-Development-Engineer_R01384-1",
    "https://synechron.wd1.myworkdayjobs.com/en-US/SynechronCareers/job/Senior-Java-Backend-Developer---Microservices---Cloud-Integration_JR1035977",
]

# Number of tracked content fields in JobExtract.
_TOTAL_CONTENT_FIELDS: int = 8

# Console table column widths
_COL_URL: int = 55
_COL_STRATEGY: int = 24
_COL_FIELDS: int = 10
_COL_TIME: int = 10
_COL_STATUS: int = 8

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Private helpers — trial execution
# ---------------------------------------------------------------------------


def _run_single_trial(
    browser,
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
        logger.error(
            "ScraperError for %s via %s: %s",
            url,
            method,
            exc.reason,
            exc_info=True,
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
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error for %s via %s: %s",
            url,
            method,
            exc,
            exc_info=True,
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


# ---------------------------------------------------------------------------
# Private helpers — persistence
# ---------------------------------------------------------------------------


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
def save_json_report(report: ExperimentReport, output_path: Path) -> None:
    """Write the experiment report to a JSON file.

    Args:
        report: Completed ``ExperimentReport``.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_report(report)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("JSON results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Private helpers — console display
# ---------------------------------------------------------------------------


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
            str(result.extract.scrape_time_ms) if result.is_success else "—"
        )
        row = (
            f"{_truncate(result.url, _COL_URL):<{_COL_URL}} | "
            f"{str(result.method):<{_COL_STRATEGY}} | "
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
    """Run all three strategies against every URL.

    Launches a single headless Chromium instance shared across all trials.
    Each trial opens and closes its own page tab.

    Args:
        job_urls: List of public job-posting URLs to scrape.

    Returns:
        A completed ``ExperimentReport`` with results for every trial.
    """
    run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
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
    """Configure logging, run the experiment, persist results, print table."""
    run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
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
