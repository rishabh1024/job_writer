"""Unit tests for agentql_job_scraper module.

All tests run without a browser or an AgentQL API key.  They cover:

- Module-level constants (query syntax, enum values)
- Pure data-transformation helpers (_flatten_context_response,
  _parse_aql_response, _count_populated_fields)
- Data models (JobExtract defaults, ScraperError attributes)
"""

from __future__ import annotations

import re

import pytest

from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    JOB_DESCRIPTION_PROMPT,
    JOB_DESCRIPTION_QUERY,
    JOB_DESCRIPTION_QUERY_WITH_CONTEXT,
    ExtractionMethod,
    JobExtract,
    ScraperError,
    _count_populated_fields,
    _flatten_context_response,
    _parse_aql_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_context_response() -> dict:
    """Simulated AgentQL response for AQL_WITH_CONTEXT query."""
    return {
        "job_title": "Staff Engineer",
        "company_name": "Acme Corp",
        "job_location": "Remote",
        "employment_type": "Full-time",
        "salary_range": "120k-160k",
        "remote_policy": "Remote",
        "application_deadline": None,
        "job_description_section": {
            "job_summary": "Build great things.",
            "responsibilities": ["Design systems", "Review code"],
            "requirements": ["5 years exp"],
            "preferred_qualifications": ["PhD"],
            "benefits": ["Health", "401k"],
        },
    }


@pytest.fixture()
def raw_flat_response() -> dict:
    """Simulated AgentQL response for AQL_STRUCTURED query."""
    return {
        "job_title": "Backend Engineer",
        "company_name": "Beta Inc",
        "job_location": "NYC",
        "employment_type": None,
        "salary_range": None,
        "job_summary": "Maintain APIs.",
        "responsibilities": ["Write code"],
        "requirements": [],
        "preferred_qualifications": [],
        "benefits": [],
        "application_deadline": None,
        "remote_policy": None,
    }


# ---------------------------------------------------------------------------
# ExtractionMethod
# ---------------------------------------------------------------------------


class TestExtractionMethod:
    def test_enum_values(self) -> None:
        assert ExtractionMethod.AQL_STRUCTURED == "aql_structured"
        assert ExtractionMethod.AQL_WITH_CONTEXT == "aql_with_context"
        assert ExtractionMethod.PROMPT_EXPERIMENTAL == "prompt_experimental"

    def test_three_methods_defined(self) -> None:
        assert len(ExtractionMethod) == 3


# ---------------------------------------------------------------------------
# Query constants
# ---------------------------------------------------------------------------


class TestQueryConstants:
    @pytest.mark.parametrize(
        "name,query",
        [
            ("BASIC", JOB_DESCRIPTION_QUERY),
            ("WITH_CONTEXT", JOB_DESCRIPTION_QUERY_WITH_CONTEXT),
        ],
    )
    def test_has_braces(self, name: str, query: str) -> None:
        assert "{" in query, f"{name}: missing opening brace"
        assert "}" in query, f"{name}: missing closing brace"

    @pytest.mark.parametrize(
        "name,query",
        [
            ("BASIC", JOB_DESCRIPTION_QUERY),
            ("WITH_CONTEXT", JOB_DESCRIPTION_QUERY_WITH_CONTEXT),
        ],
    )
    def test_balanced_braces(self, name: str, query: str) -> None:
        assert query.count("{") == query.count("}"), (
            f"{name}: unbalanced braces"
        )

    @pytest.mark.parametrize(
        "name,query",
        [
            ("BASIC", JOB_DESCRIPTION_QUERY),
            ("WITH_CONTEXT", JOB_DESCRIPTION_QUERY_WITH_CONTEXT),
        ],
    )
    def test_no_multiline_semantic_context(
        self, name: str, query: str
    ) -> None:
        """AgentQL does not allow newlines inside parenthesised context."""
        matches = re.findall(r"\([^)]*\n[^)]*\)", query)
        assert matches == [], (
            f"{name}: multi-line semantic context found: {matches}"
        )

    def test_context_query_has_semantic_hints(self) -> None:
        for hint in [
            "job_title(",
            "company_name(",
            "job_location(",
            "responsibilities(",
            "requirements(",
            "benefits(",
        ]:
            assert hint in JOB_DESCRIPTION_QUERY_WITH_CONTEXT, (
                f"Missing semantic hint: {hint}"
            )

    def test_context_query_has_structural_nesting(self) -> None:
        assert "job_description_section" in JOB_DESCRIPTION_QUERY_WITH_CONTEXT

    def test_prompt_is_non_empty_string(self) -> None:
        assert isinstance(JOB_DESCRIPTION_PROMPT, str)
        assert len(JOB_DESCRIPTION_PROMPT) > 20


# ---------------------------------------------------------------------------
# _flatten_context_response
# ---------------------------------------------------------------------------


class TestFlattenContextResponse:
    def test_hoists_nested_fields(
        self, raw_context_response: dict
    ) -> None:
        flat = _flatten_context_response(raw_context_response)
        assert flat["job_summary"] == "Build great things."
        assert flat["responsibilities"] == ["Design systems", "Review code"]
        assert flat["requirements"] == ["5 years exp"]
        assert flat["preferred_qualifications"] == ["PhD"]
        assert flat["benefits"] == ["Health", "401k"]

    def test_preserves_top_level_fields(
        self, raw_context_response: dict
    ) -> None:
        flat = _flatten_context_response(raw_context_response)
        assert flat["job_title"] == "Staff Engineer"
        assert flat["company_name"] == "Acme Corp"
        assert flat["employment_type"] == "Full-time"
        assert flat["salary_range"] == "120k-160k"

    def test_missing_section_returns_none_for_nested(self) -> None:
        flat = _flatten_context_response({"job_title": "PM"})
        assert flat["job_title"] == "PM"
        assert flat["job_summary"] is None
        assert flat["responsibilities"] is None

    def test_empty_dict_returns_all_none(self) -> None:
        flat = _flatten_context_response({})
        for key in [
            "job_title",
            "job_summary",
            "responsibilities",
            "requirements",
        ]:
            assert flat[key] is None


# ---------------------------------------------------------------------------
# _parse_aql_response
# ---------------------------------------------------------------------------


class TestParseAqlResponse:
    def test_aql_with_context_full_result(
        self, raw_context_response: dict
    ) -> None:
        extract = _parse_aql_response(
            raw_context_response,
            "https://example.com/job",
            ExtractionMethod.AQL_WITH_CONTEXT,
        )
        assert extract.job_title == "Staff Engineer"
        assert extract.job_summary == "Build great things."
        assert extract.responsibilities == ["Design systems", "Review code"]
        # 11 of 12 fields populated (application_deadline is None)
        assert extract.populated_fields == 11

    def test_aql_structured_flat_result(
        self, raw_flat_response: dict
    ) -> None:
        extract = _parse_aql_response(
            raw_flat_response,
            "https://example.com/job2",
            ExtractionMethod.AQL_STRUCTURED,
        )
        assert extract.job_title == "Backend Engineer"
        # title, company, location, summary, responsibilities -> 5
        assert extract.populated_fields == 5

    def test_prompt_experimental_flat_result(self) -> None:
        raw = {
            "job_title": "PM",
            "responsibilities": ["Define roadmap"],
        }
        extract = _parse_aql_response(
            raw, "https://x.com", ExtractionMethod.PROMPT_EXPERIMENTAL
        )
        assert extract.job_title == "PM"
        assert extract.responsibilities == ["Define roadmap"]
        assert extract.populated_fields == 2

    def test_aql_with_context_missing_section_graceful(self) -> None:
        extract = _parse_aql_response(
            {"job_title": "PM"},
            "https://x.com",
            ExtractionMethod.AQL_WITH_CONTEXT,
        )
        assert extract.responsibilities == []
        assert extract.populated_fields == 1

    def test_none_lists_default_to_empty(self) -> None:
        raw = {
            "job_title": "Engineer",
            "responsibilities": None,
            "requirements": None,
        }
        extract = _parse_aql_response(
            raw, "https://x.com", ExtractionMethod.AQL_STRUCTURED
        )
        assert extract.responsibilities == []
        assert extract.requirements == []

    def test_url_and_method_stamped(self, raw_flat_response: dict) -> None:
        url = "https://example.com/job"
        method = ExtractionMethod.AQL_STRUCTURED
        extract = _parse_aql_response(raw_flat_response, url, method)
        assert extract.url == url
        assert extract.method == method


# ---------------------------------------------------------------------------
# _count_populated_fields
# ---------------------------------------------------------------------------


class TestCountPopulatedFields:
    def test_all_empty(self) -> None:
        j = JobExtract(url="x", method=ExtractionMethod.AQL_STRUCTURED)
        assert _count_populated_fields(j) == 0

    def test_all_populated(self) -> None:
        j = JobExtract(
            url="x",
            method=ExtractionMethod.AQL_STRUCTURED,
            job_title="T",
            company_name="C",
            job_location="L",
            employment_type="FT",
            salary_range="100k",
            job_summary="S",
            responsibilities=["R"],
            requirements=["Q"],
            preferred_qualifications=["P"],
            benefits=["B"],
            application_deadline="2026-12-01",
            remote_policy="Remote",
        )
        assert _count_populated_fields(j) == 12

    def test_partial(self) -> None:
        j = JobExtract(
            url="x",
            method=ExtractionMethod.AQL_STRUCTURED,
            job_title="T",
            company_name="C",
        )
        assert _count_populated_fields(j) == 2


# ---------------------------------------------------------------------------
# ScraperError
# ---------------------------------------------------------------------------


class TestScraperError:
    def test_attributes(self) -> None:
        err = ScraperError("https://example.com", "timeout")
        assert err.url == "https://example.com"
        assert err.reason == "timeout"

    def test_is_exception(self) -> None:
        err = ScraperError("https://x.com", "bad")
        assert isinstance(err, Exception)

    def test_str_contains_reason(self) -> None:
        err = ScraperError("https://x.com", "connection refused")
        assert "connection refused" in str(err)

    def test_str_contains_url(self) -> None:
        err = ScraperError("https://x.com", "fail")
        assert "https://x.com" in str(err)


# ---------------------------------------------------------------------------
# JobExtract defaults
# ---------------------------------------------------------------------------


class TestJobExtractDefaults:
    def test_defaults(self) -> None:
        j = JobExtract(url="https://x.com", method=ExtractionMethod.AQL_STRUCTURED)
        assert j.has_error is False
        assert j.responsibilities == []
        assert j.requirements == []
        assert j.preferred_qualifications == []
        assert j.benefits == []
        assert j.populated_fields == 0
        assert j.scrape_time_ms == 0
        assert j.error_message is None

    def test_content_fields_covers_12(self) -> None:
        j = JobExtract(url="x", method=ExtractionMethod.AQL_STRUCTURED)
        assert len(j._CONTENT_FIELDS) == 12

    def test_content_fields_match_expected_set(self) -> None:
        j = JobExtract(url="x", method=ExtractionMethod.AQL_STRUCTURED)
        expected = {
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
        }
        assert set(j._CONTENT_FIELDS) == expected
