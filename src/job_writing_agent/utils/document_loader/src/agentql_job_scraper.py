"""AgentQL-powered job description scraper.

This module provides a single public function, ``extract_job_data``, that uses
the AgentQL Playwright SDK to extract structured job-description fields from any
public job-posting URL.  It raises typed exceptions on failure so the caller can
decide how to handle errors gracefully.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

import agentql
from agentql.ext.playwright.sync_api import Page
from playwright.sync_api import Browser

from job_writing_agent.utils.app_log.logging_config import get_logger
from job_writing_agent.utils.app_log.logging_decorators import log_execution

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_TIMEOUT_MS: int = 30_000
SLOW_RESPONSE_THRESHOLD_S: float = 15.0

# Structured AgentQL query covering the standard fields found on most job
# boards.  All fields are optional from the scraper's perspective — AgentQL
# returns None for fields it cannot locate on the page.
JOB_DESCRIPTION_QUERY: str = """
{
    job_title
    company_name
    job_location
    employment_type
    salary_range
    job_summary
    responsibilities[]
    requirements[]
    preferred_qualifications[]
    benefits[]
    application_deadline
    remote_policy
}
"""

# Natural-language fallback prompt used when the structured AQL query returns
# mostly empty fields (e.g. heavily JavaScript-rendered boards).
JOB_DESCRIPTION_NL_PROMPT: str = (
    "Extract all job description details including job title, company name, "
    "location, employment type, salary, summary, responsibilities, "
    "requirements, preferred qualifications, benefits, and deadline."
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ScraperError(Exception):
    """Raised when the scraper cannot extract data from a page.

    Attributes:
        url: The URL that failed.
        reason: Human-readable description of the failure.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(f"Scrape failed for {url!r}: {reason}")


class ExtractionMethod(StrEnum):
    """Supported extraction strategies."""

    AQL_STRUCTURED = "aql_structured"
    AQL_NATURAL_LANGUAGE = "aql_natural_language"


@dataclass
class JobExtract:
    """Structured output of a single job-description extraction.

    All content fields default to ``None`` when AgentQL could not locate them
    on the page.  ``has_error`` is ``True`` only when the entire extraction
    failed; partial results (some fields present) are still returned with
    ``has_error=False``.

    Attributes:
        url: Source URL of the job posting.
        method: Extraction strategy that produced this result.
        job_title: Title of the role.
        company_name: Hiring company.
        job_location: Office / city / remote label.
        employment_type: Full-time, part-time, contract, etc.
        salary_range: Compensation range as a string.
        job_summary: Introductory paragraph or overview.
        responsibilities: List of role responsibilities.
        requirements: Mandatory qualifications.
        preferred_qualifications: Nice-to-have qualifications.
        benefits: Perks and benefits listed.
        application_deadline: Closing date for applications.
        remote_policy: Remote / hybrid / on-site policy.
        scrape_time_ms: Wall-clock time for the extraction in milliseconds.
        populated_fields: Count of non-None content fields extracted.
        has_error: Whether extraction raised an exception.
        error_message: Exception message when ``has_error`` is ``True``.
    """

    url: str
    method: ExtractionMethod

    job_title: Optional[str] = None
    company_name: Optional[str] = None
    job_location: Optional[str] = None
    employment_type: Optional[str] = None
    salary_range: Optional[str] = None
    job_summary: Optional[str] = None
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    preferred_qualifications: list[str] = field(default_factory=list)
    benefits: list[str] = field(default_factory=list)
    application_deadline: Optional[str] = None
    remote_policy: Optional[str] = None

    scrape_time_ms: int = 0
    populated_fields: int = 0
    has_error: bool = False
    error_message: Optional[str] = None

    # Total content fields checked by ``_count_populated_fields``
    _CONTENT_FIELDS: tuple[str, ...] = field(
        default=(
            "job_title",
            "company_name",
            "job_location",
            "employment_type",
            "salary_range",
            "job_summary",
            "responsibilities",
            "requirements",
            "preferred_qualifications",
            "benefits",
            "application_deadline",
            "remote_policy",
        ),
        init=False,
        repr=False,
        compare=False,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _navigate_to_url(page: Page, url: str) -> None:
    """Navigate the browser page to ``url`` and wait for load.

    Args:
        page: AgentQL-wrapped Playwright page.
        url: Target URL to navigate to.

    Raises:
        ScraperError: If navigation times out or the network is unreachable.
    """
    logger.debug("Navigating to %s", url)
    try:
        page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
    except Exception as exc:
        raise ScraperError(url, f"Navigation failed: {exc}") from exc


def _parse_aql_response(
    raw: dict, url: str, method: ExtractionMethod
) -> JobExtract:
    """Convert a raw AgentQL ``query_data`` dict into a ``JobExtract``.

    Args:
        raw: Dictionary returned by ``page.query_data()``.
        url: Source URL (stored in the result for traceability).
        method: Extraction method tag to stamp on the result.

    Returns:
        A populated ``JobExtract`` instance.
    """
    logger.debug("Raw AgentQL response for %s: %s", url, raw)

    extract = JobExtract(
        url=url,
        method=method,
        job_title=raw.get("job_title"),
        company_name=raw.get("company_name"),
        job_location=raw.get("job_location"),
        employment_type=raw.get("employment_type"),
        salary_range=raw.get("salary_range"),
        job_summary=raw.get("job_summary"),
        responsibilities=raw.get("responsibilities") or [],
        requirements=raw.get("requirements") or [],
        preferred_qualifications=raw.get("preferred_qualifications") or [],
        benefits=raw.get("benefits") or [],
        application_deadline=raw.get("application_deadline"),
        remote_policy=raw.get("remote_policy"),
    )
    extract.populated_fields = _count_populated_fields(extract)
    return extract


def _count_populated_fields(extract: JobExtract) -> int:
    """Count how many content fields contain non-empty values.

    Args:
        extract: The ``JobExtract`` to inspect.

    Returns:
        Integer count of fields that are truthy (non-None, non-empty list).
    """
    count = 0
    for field_name in extract._CONTENT_FIELDS:
        value = getattr(extract, field_name)
        if value:
            count += 1
    return count


def _warn_if_partial(extract: JobExtract, total_fields: int) -> None:
    """Emit a WARNING when fewer than half of content fields were populated.

    Args:
        extract: Completed ``JobExtract``.
        total_fields: Total number of tracked content fields.
    """
    if extract.populated_fields < total_fields // 2:
        logger.warning(
            "Partial extraction for %s via %s: only %d/%d fields populated",
            extract.url,
            extract.method,
            extract.populated_fields,
            total_fields,
        )


def _warn_if_slow(elapsed_s: float, url: str, method: ExtractionMethod) -> None:
    """Emit a WARNING when a scrape exceeds the slow-response threshold.

    Args:
        elapsed_s: Total elapsed seconds for the scrape.
        url: Source URL.
        method: Extraction method used.
    """
    if elapsed_s > SLOW_RESPONSE_THRESHOLD_S:
        logger.warning(
            "Slow response for %s via %s: %.1fs (threshold %.0fs)",
            url,
            method,
            elapsed_s,
            SLOW_RESPONSE_THRESHOLD_S,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@log_execution
def extract_job_data(
    browser: Browser,
    url: str,
    method: ExtractionMethod = ExtractionMethod.AQL_STRUCTURED,
) -> JobExtract:
    """Extract structured job-description data from a public job posting URL.

    Opens a new browser page, navigates to ``url``, and calls
    ``page.query_data()`` using the appropriate AgentQL strategy.  The page is
    always closed after extraction regardless of success or failure.

    Args:
        browser: An already-launched Playwright ``Browser`` instance.
        url: Publicly accessible URL of the job posting.
        method: Extraction strategy to use.  Defaults to
            ``ExtractionMethod.AQL_STRUCTURED``.

    Returns:
        A ``JobExtract`` dataclass with all discovered fields populated.

    Raises:
        ScraperError: If navigation fails or AgentQL raises an unexpected
            exception during data extraction.
    """
    logger.info("Starting extraction: url=%s method=%s", url, method)
    start_time = time.monotonic()

    page: Page = agentql.wrap(browser.new_page())
    try:
        _navigate_to_url(page, url)

        if method == ExtractionMethod.AQL_STRUCTURED:
            raw: dict = page.query_data(JOB_DESCRIPTION_QUERY)
        else:
            raw = page.query_data(JOB_DESCRIPTION_NL_PROMPT)

        elapsed_s = time.monotonic() - start_time
        extract = _parse_aql_response(raw, url, method)
        extract.scrape_time_ms = int(elapsed_s * 1_000)

        _warn_if_slow(elapsed_s, url, method)
        _warn_if_partial(extract, len(extract._CONTENT_FIELDS))

        logger.info(
            "Extraction complete: url=%s method=%s fields=%d/%d time=%dms",
            url,
            method,
            extract.populated_fields,
            len(extract._CONTENT_FIELDS),
            extract.scrape_time_ms,
        )
        return extract

    except ScraperError:
        raise
    except Exception as exc:
        elapsed_s = time.monotonic() - start_time
        raise ScraperError(url, str(exc)) from exc
    finally:
        page.close()
