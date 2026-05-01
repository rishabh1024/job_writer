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

# ---------------------------------------------------------------------------
# Bare AQL query — field names only, no context hints.
# Used as the Method A baseline so results can be compared against the
# context-enriched variant below.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Context-enriched AQL query — applies both AgentQL best practices:
#
#   1. Semantic context: parentheses descriptions on every field guide AgentQL
#      to the correct element when multiple candidates share similar text.
#      e.g.  job_title(the h1 or prominent heading naming the role)
#
#   2. Structural context: list fields that belong to the same prose section
#      are nested under a job_description_section container, mirroring the
#      typical DOM hierarchy of a job-posting page and reducing ambiguity
#      with unrelated lists (nav links, footer items, etc.).
#
# Reference: https://docs.agentql.com/agentql-query/best-practices
# ---------------------------------------------------------------------------
JOB_DESCRIPTION_QUERY_WITH_CONTEXT: str = """
{
    job_title(the h1 or prominent heading that names the open role)
    company_name(the name of the hiring organisation or employer)
    job_location(office city, region, country or remote label for the role)
    employment_type(full-time, part-time, contract or internship label)
    salary_range(compensation, pay or salary range shown on the posting)
    remote_policy(remote, hybrid or on-site work arrangement for the role)
    application_deadline(closing date or apply-by date for the role)
    job_description_section(the main body section of the job posting) {
        job_summary(introductory paragraph or overview of the role)
        responsibilities(list of duties and day-to-day tasks for the role)[]
        requirements(mandatory qualifications, skills or experience needed)[]
        preferred_qualifications(nice-to-have or bonus qualifications)[]
        benefits(perks, compensation extras or employee benefits listed)[]
    }
}
"""

# Prompt for the experimental prompt-based extraction method.
# Passed to get_data_by_prompt_experimental() which uses a different
# (non-AQL) inference path on the AgentQL backend.
JOB_DESCRIPTION_PROMPT: str = (
    "Extract all job description details: job title, company name, "
    "location, employment type, salary range, remote policy, "
    "application deadline, job summary, list of responsibilities, "
    "list of requirements, list of preferred qualifications, "
    "and list of benefits."
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
    """Supported extraction strategies.

    Attributes:
        AQL_STRUCTURED: Bare AQL query with field names only.  Useful as a
            baseline to measure the uplift from adding context hints.
        AQL_WITH_CONTEXT: AQL query enriched with semantic context
            descriptions ``(...)`` on every field and structural nesting for
            the main description section.  This is the recommended approach
            per AgentQL best practices.
        PROMPT_EXPERIMENTAL: Free-form natural-language prompt passed to
            ``get_data_by_prompt_experimental()``.  Uses a different AgentQL
            inference path that does not require an AQL query.
    """

    AQL_STRUCTURED = "aql_structured"
    AQL_WITH_CONTEXT = "aql_with_context"
    PROMPT_EXPERIMENTAL = "prompt_experimental"


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


def _flatten_context_response(raw: dict) -> dict:
    """Hoist nested ``job_description_section`` fields up to the top level.

    ``JOB_DESCRIPTION_QUERY_WITH_CONTEXT`` nests ``job_summary``,
    ``responsibilities``, ``requirements``, ``preferred_qualifications``, and
    ``benefits`` inside a ``job_description_section`` container.  This helper
    merges those fields back into a flat dict so ``_parse_aql_response`` can
    use a single code path regardless of which query was used.

    Args:
        raw: Dictionary returned by ``page.query_data()`` when the context
            query was used.

    Returns:
        A new flat dict with all top-level and nested content fields merged.
    """
    section: dict = raw.get("job_description_section") or {}
    return {
        "job_title": raw.get("job_title"),
        "company_name": raw.get("company_name"),
        "job_location": raw.get("job_location"),
        "employment_type": raw.get("employment_type"),
        "salary_range": raw.get("salary_range"),
        "remote_policy": raw.get("remote_policy"),
        "application_deadline": raw.get("application_deadline"),
        "job_summary": section.get("job_summary"),
        "responsibilities": section.get("responsibilities"),
        "requirements": section.get("requirements"),
        "preferred_qualifications": section.get("preferred_qualifications"),
        "benefits": section.get("benefits"),
    }


def _parse_aql_response(
    raw: dict, url: str, method: ExtractionMethod
) -> JobExtract:
    """Convert a raw AgentQL ``query_data`` dict into a ``JobExtract``.

    Handles both flat (``AQL_STRUCTURED`` / ``PROMPT_EXPERIMENTAL``) and
    nested (``AQL_WITH_CONTEXT``) response shapes transparently.

    Args:
        raw: Dictionary returned by ``page.query_data()`` or
            ``page.get_data_by_prompt_experimental()``.
        url: Source URL (stored in the result for traceability).
        method: Extraction method tag to stamp on the result.

    Returns:
        A populated ``JobExtract`` instance.
    """
    logger.debug("Raw AgentQL response for %s: %s", url, raw)

    if method == ExtractionMethod.AQL_WITH_CONTEXT:
        flat = _flatten_context_response(raw)
    else:
        flat = raw

    extract = JobExtract(
        url=url,
        method=method,
        job_title=flat.get("job_title"),
        company_name=flat.get("company_name"),
        job_location=flat.get("job_location"),
        employment_type=flat.get("employment_type"),
        salary_range=flat.get("salary_range"),
        job_summary=flat.get("job_summary"),
        responsibilities=flat.get("responsibilities") or [],
        requirements=flat.get("requirements") or [],
        preferred_qualifications=flat.get("preferred_qualifications") or [],
        benefits=flat.get("benefits") or [],
        application_deadline=flat.get("application_deadline"),
        remote_policy=flat.get("remote_policy"),
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
        elif method == ExtractionMethod.AQL_WITH_CONTEXT:
            raw = page.query_data(JOB_DESCRIPTION_QUERY_WITH_CONTEXT)
        else:
            # PROMPT_EXPERIMENTAL uses a separate AgentQL inference path
            # that accepts free-form natural language instead of an AQL query.
            raw = page.get_data_by_prompt_experimental(
                JOB_DESCRIPTION_PROMPT
            )

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
