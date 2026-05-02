"""AgentQL-powered job description scraper.

Public API
----------
``AgentQlJobScraper``
    Class that accepts any ``BaseScraperStrategy``, manages page lifecycle,
    and returns a ``JobExtract``.  Preferred entry point.

``extract_job_data``
    Backward-compatible function shim retained for existing callers and tests.

Data models
-----------
``JobExtract``    — structured extraction result dataclass
``ScraperError``  — typed exception raised on navigation / extraction failure
``ExtractionMethod`` — StrEnum identifying each strategy variant

Query / prompt constants
------------------------
``JOB_DESCRIPTION_QUERY``              — bare AQL query (no context)
``JOB_DESCRIPTION_QUERY_WITH_CONTEXT`` — AQL query with semantic + structural context
``JOB_DESCRIPTION_PROMPT``             — free-form NL prompt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

import agentql
from agentql.ext.playwright.sync_api import Page
from playwright.sync_api import Browser

from job_writing_agent.utils.app_log.logging_config import get_logger
from job_writing_agent.utils.app_log.logging_decorators import log_execution

if TYPE_CHECKING:
    from job_writing_agent.utils.document_loader.src.strategies.base import (
        BaseScraperStrategy,
    )

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_TIMEOUT_MS: int = 30_000
SLOW_RESPONSE_THRESHOLD_S: float = 15.0

# ---------------------------------------------------------------------------
# Bare AQL query — field names only, no context hints.
# Kept here (not in the strategy file) so tests and the function shim can
# import it without importing the strategy sub-package.
# ---------------------------------------------------------------------------
JOB_DESCRIPTION_QUERY: str = """
{
    body[] {
    job_title
    company_name
    job_location
    job_summary
    responsibilities[]
    requirements[]
    preferred_qualifications[]
    benefits[]
        }
}
"""

# ---------------------------------------------------------------------------
# Context-enriched AQL query.
# Applies both AgentQL best practices:
#   1. Semantic context — parentheses descriptions on every field
#   2. Structural context — list fields nested under job_description_section
# Reference: https://docs.agentql.com/agentql-query/best-practices
# ---------------------------------------------------------------------------
JOB_DESCRIPTION_QUERY_WITH_CONTEXT: str = """
{
    body[] {
        job_title(the h1 or prominent heading that names the open role)
    company_name(the name of the hiring organisation or employer)
    job_location(office city, region, country or remote label for the role)
    job_description_section(the main body section of the job posting) {
        job_summary(introductory paragraph or overview of the role)
        responsibilities(list of duties and day-to-day tasks for the role)[]
        requirements(mandatory qualifications, skills or experience needed)[]
        preferred_qualifications(nice-to-have or bonus qualifications)[]
        benefits(perks, compensation extras or employee benefits listed)[]
    }
    }
}
"""

# Prompt for the experimental prompt-based extraction method.
# Passed to get_data_by_prompt_experimental() which uses a different
# (non-AQL) inference path on the AgentQL backend.
JOB_DESCRIPTION_PROMPT: str = (
    "Extract all job description details: job title, company name, "
    "location, job summary, list of responsibilities, list of requirements, "
    "list of preferred qualifications, and list of benefits."
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
        job_summary: Introductory paragraph or overview.
        responsibilities: List of role responsibilities.
        requirements: Mandatory qualifications.
        preferred_qualifications: Nice-to-have qualifications.
        benefits: Perks and benefits listed.
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
    job_summary: Optional[str] = None
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    preferred_qualifications: list[str] = field(default_factory=list)
    benefits: list[str] = field(default_factory=list)

    scrape_time_ms: int = 0
    populated_fields: int = 0
    has_error: bool = False
    error_message: Optional[str] = None

    _CONTENT_FIELDS: tuple[str, ...] = field(
        default=(
            "job_title",
            "company_name",
            "job_location",
            "job_summary",
            "responsibilities",
            "requirements",
            "preferred_qualifications",
            "benefits",
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


def _is_present(value: object) -> bool:
    """Return whether a raw response value contains usable content."""
    if isinstance(value, list):
        return any(_is_present(item) for item in value)
    if isinstance(value, dict):
        return any(_is_present(item) for item in value.values())
    return bool(value)


def _score_body_candidate(candidate: dict) -> int:
    """Score a ``body[]`` candidate by how many target fields it contains."""
    section = candidate.get("job_description_section") or {}
    score = 0
    for field_name in (
        "job_title",
        "company_name",
        "job_location",
        "job_summary",
        "responsibilities",
        "requirements",
        "preferred_qualifications",
        "benefits",
    ):
        value = candidate.get(field_name)
        if value is None and isinstance(section, dict):
            value = section.get(field_name)
        if _is_present(value):
            score += 1
    return score


def _unwrap_body_response(raw: dict) -> dict:
    """Return the best extraction payload from a top-level ``body`` wrapper.

    Queries that use ``body[]`` return ``{"body": [{...}]}`` rather than a
    flat field dictionary.  Older queries returned fields at the top level, so
    this helper accepts both shapes.
    """
    body = raw.get("body")
    if isinstance(body, dict):
        return body
    if isinstance(body, list):
        candidates = [item for item in body if isinstance(item, dict)]
        if candidates:
            return max(candidates, key=_score_body_candidate)
    return raw


def _flatten_context_response(raw: dict) -> dict:
    """Hoist nested ``job_description_section`` fields up to the top level.

    ``JOB_DESCRIPTION_QUERY_WITH_CONTEXT`` wraps fields in ``body[]`` and
    nests ``job_summary``, ``responsibilities``, ``requirements``,
    ``preferred_qualifications``, and ``benefits`` inside a
    ``job_description_section`` container.  This helper merges those fields
    back into a flat dict so ``_parse_aql_response`` can use a single code path
    regardless of which query was used.

    Args:
        raw: Dictionary returned by ``page.query_data()`` when the context
            query was used.

    Returns:
        A new flat dict with all top-level and nested content fields merged.
    """
    body = _unwrap_body_response(raw)
    section_raw = body.get("job_description_section") or {}
    section = section_raw if isinstance(section_raw, dict) else {}
    return {
        "job_title": body.get("job_title"),
        "company_name": body.get("company_name"),
        "job_location": body.get("job_location"),
        "job_summary": section.get("job_summary") or body.get("job_summary"),
        "responsibilities": section.get("responsibilities")
        or body.get("responsibilities"),
        "requirements": section.get("requirements") or body.get("requirements"),
        "preferred_qualifications": section.get("preferred_qualifications")
        or body.get("preferred_qualifications"),
        "benefits": section.get("benefits") or body.get("benefits"),
    }


def _parse_aql_response(
    raw: dict, url: str, method: ExtractionMethod
) -> JobExtract:
    """Convert a raw AgentQL response dict into a ``JobExtract``.

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
        flat = _unwrap_body_response(raw)

    extract = JobExtract(
        url=url,
        method=method,
        job_title=flat.get("job_title"),
        company_name=flat.get("company_name"),
        job_location=flat.get("job_location"),
        job_summary=flat.get("job_summary"),
        responsibilities=flat.get("responsibilities") or [],
        requirements=flat.get("requirements") or [],
        preferred_qualifications=flat.get("preferred_qualifications") or [],
        benefits=flat.get("benefits") or [],
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
# AgentQlJobScraper — strategy-pattern entry point
# ---------------------------------------------------------------------------


class AgentQlJobScraper:
    """Scrapes a job-posting URL using an interchangeable extraction strategy.

    Decouples page lifecycle management (navigation, timeout, page close) from
    the extraction logic, which is entirely owned by the injected
    ``BaseScraperStrategy``.  Swap the strategy without changing this class.

    Example:
        >>> from strategies import AqlWithContextStrategy
        >>> scraper = AgentQlJobScraper(browser, AqlWithContextStrategy())
        >>> result = scraper.scrape("https://example.com/job/123")

    Args:
        browser: An already-launched Playwright ``Browser`` instance.
        strategy: Concrete ``BaseScraperStrategy`` to use for extraction.
    """

    def __init__(self, browser: Browser, strategy: BaseScraperStrategy) -> None:
        self._browser = browser
        self._strategy = strategy

    @property
    def strategy(self) -> BaseScraperStrategy:
        """The currently active extraction strategy.

        Returns:
            The ``BaseScraperStrategy`` instance injected at construction.
        """
        return self._strategy

    @log_execution
    def scrape(self, url: str) -> JobExtract:
        """Extract job-description data from ``url`` using the active strategy.

        Opens a new browser tab, navigates to ``url``, delegates extraction to
        ``self._strategy.execute()``, then closes the tab.  The tab is always
        closed in a ``finally`` block regardless of success or failure.

        Args:
            url: Publicly accessible job-posting URL.

        Returns:
            A ``JobExtract`` with all discovered fields populated.

        Raises:
            ScraperError: If navigation fails or the strategy raises an
                unexpected exception.
        """
        method = self._strategy.method_name
        logger.info("Starting extraction: url=%s strategy=%s", url, method)
        start_time = time.monotonic()

        page: Page = agentql.wrap(self._browser.new_page())
        try:
            _navigate_to_url(page, url)
            raw: dict = self._strategy.execute(page)

            elapsed_s = time.monotonic() - start_time
            extract = _parse_aql_response(raw, url, method)
            extract.scrape_time_ms = int(elapsed_s * 1_000)

            _warn_if_slow(elapsed_s, url, method)
            _warn_if_partial(extract, len(extract._CONTENT_FIELDS))

            logger.info(
                "Extraction complete: url=%s strategy=%s "
                "fields=%d/%d time=%dms",
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


# ---------------------------------------------------------------------------
# Backward-compatible function shim
# ---------------------------------------------------------------------------


@log_execution
def extract_job_data(
    browser: Browser,
    url: str,
    method: ExtractionMethod = ExtractionMethod.AQL_STRUCTURED,
) -> JobExtract:
    """Extract job-description data using the given ``ExtractionMethod``.

    This function is a thin shim around ``AgentQlJobScraper`` that maps the
    ``ExtractionMethod`` enum to the appropriate strategy class.  It exists
    for backward compatibility with existing callers and tests.

    Prefer instantiating ``AgentQlJobScraper`` directly with an explicit
    strategy for new code.

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
        ValueError: If ``method`` is not a recognised ``ExtractionMethod``.
    """
    from job_writing_agent.utils.document_loader.src.strategies import (
        AqlStructuredStrategy,
        AqlWithContextStrategy,
        PromptExperimentalStrategy,
    )

    strategy_map = {
        ExtractionMethod.AQL_STRUCTURED: AqlStructuredStrategy,
        ExtractionMethod.AQL_WITH_CONTEXT: AqlWithContextStrategy,
        ExtractionMethod.PROMPT_EXPERIMENTAL: PromptExperimentalStrategy,
    }

    strategy_cls = strategy_map.get(method)
    if strategy_cls is None:
        raise ValueError(f"Unrecognised ExtractionMethod: {method!r}")

    scraper = AgentQlJobScraper(browser, strategy_cls())
    return scraper.scrape(url)
