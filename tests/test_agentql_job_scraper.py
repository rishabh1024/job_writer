"""Unit tests for agentql_job_scraper module."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    JOB_DESCRIPTION_QUERY_WITH_CONTEXT,
    AgentQlJobScraper,
    ExtractionMethod,
    JobExtract,
    ScraperError,
    _count_populated_fields,
    _flatten_context_response,
    _parse_aql_response,
)
from job_writing_agent.utils.document_loader.src.strategies import (
    AqlWithContextStrategy,
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
        "job_description_section": {
            "job_summary": "Build great things.",
            "responsibilities": ["Design systems", "Review code"],
            "requirements": ["5 years exp"],
            "preferred_qualifications": ["PhD"],
            "benefits": ["Health", "401k"],
        },
    }


# ---------------------------------------------------------------------------
# ExtractionMethod
# ---------------------------------------------------------------------------


class TestExtractionMethod:
    def test_enum_values(self) -> None:
        assert ExtractionMethod.AQL_WITH_CONTEXT == "aql_with_context"

    def test_only_context_method_defined(self) -> None:
        assert list(ExtractionMethod) == [ExtractionMethod.AQL_WITH_CONTEXT]


# ---------------------------------------------------------------------------
# Query constants
# ---------------------------------------------------------------------------


class TestQueryConstants:
    def test_has_braces(self) -> None:
        assert "{" in JOB_DESCRIPTION_QUERY_WITH_CONTEXT
        assert "}" in JOB_DESCRIPTION_QUERY_WITH_CONTEXT

    def test_balanced_braces(self) -> None:
        assert JOB_DESCRIPTION_QUERY_WITH_CONTEXT.count(
            "{"
        ) == JOB_DESCRIPTION_QUERY_WITH_CONTEXT.count("}")

    def test_no_multiline_semantic_context(self) -> None:
        matches = re.findall(
            r"\([^)]*\n[^)]*\)",
            JOB_DESCRIPTION_QUERY_WITH_CONTEXT,
        )
        assert matches == []

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
        assert flat["job_location"] == "Remote"

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
        assert extract.populated_fields == 8

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
            raw, "https://x.com", ExtractionMethod.AQL_WITH_CONTEXT
        )
        assert extract.responsibilities == []
        assert extract.requirements == []

    def test_url_and_method_stamped(self, raw_context_response: dict) -> None:
        url = "https://example.com/job"
        method = ExtractionMethod.AQL_WITH_CONTEXT
        extract = _parse_aql_response(raw_context_response, url, method)
        assert extract.url == url
        assert extract.method == method


# ---------------------------------------------------------------------------
# _count_populated_fields
# ---------------------------------------------------------------------------


class TestCountPopulatedFields:
    def test_all_empty(self) -> None:
        j = JobExtract(url="x", method=ExtractionMethod.AQL_WITH_CONTEXT)
        assert _count_populated_fields(j) == 0

    def test_all_populated(self) -> None:
        j = JobExtract(
            url="x",
            method=ExtractionMethod.AQL_WITH_CONTEXT,
            job_title="T",
            company_name="C",
            job_location="L",
            job_summary="S",
            responsibilities=["R"],
            requirements=["Q"],
            preferred_qualifications=["P"],
            benefits=["B"],
        )
        assert _count_populated_fields(j) == 8

    def test_partial(self) -> None:
        j = JobExtract(
            url="x",
            method=ExtractionMethod.AQL_WITH_CONTEXT,
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
        assert isinstance(ScraperError("x", "y"), Exception)

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
        j = JobExtract(
            url="https://x.com", method=ExtractionMethod.AQL_WITH_CONTEXT
        )
        assert j.has_error is False
        assert j.responsibilities == []
        assert j.requirements == []
        assert j.preferred_qualifications == []
        assert j.benefits == []
        assert j.populated_fields == 0
        assert j.scrape_time_ms == 0
        assert j.error_message is None

    def test_content_fields_covers_8(self) -> None:
        assert len(JobExtract.CONTENT_FIELDS) == 8

    def test_content_fields_match_expected_set(self) -> None:
        expected = {
            "job_title",
            "company_name",
            "job_location",
            "job_summary",
            "responsibilities",
            "requirements",
            "preferred_qualifications",
            "benefits",
        }
        assert set(JobExtract.CONTENT_FIELDS) == expected


# ---------------------------------------------------------------------------
# Strategy pattern — concrete strategies
# ---------------------------------------------------------------------------


class TestConcreteStrategies:
    def test_method_name(self) -> None:
        assert (
            AqlWithContextStrategy().method_name
            == ExtractionMethod.AQL_WITH_CONTEXT
        )

    def test_description_is_non_empty_string(self) -> None:
        desc = AqlWithContextStrategy().description
        assert isinstance(desc, str) and len(desc) > 5

    def test_aql_with_context_calls_query_data(self) -> None:
        page = MagicMock()
        page.query_data.return_value = {"job_title": "Eng"}
        result = AqlWithContextStrategy().execute(page)
        page.query_data.assert_called_once_with(
            JOB_DESCRIPTION_QUERY_WITH_CONTEXT
        )
        assert result == {"job_title": "Eng"}


# ---------------------------------------------------------------------------
# AgentQlJobScraper
# ---------------------------------------------------------------------------


class TestAgentQlJobScraper:
    def test_strategy_property(self) -> None:
        strategy = AqlWithContextStrategy()
        scraper = AgentQlJobScraper(MagicMock(), strategy)
        assert scraper.strategy is strategy

    def test_scrape_delegates_to_strategy(self) -> None:
        strategy = MagicMock()
        strategy.method_name = ExtractionMethod.AQL_WITH_CONTEXT
        strategy.execute.return_value = {"job_title": "Eng"}

        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page

        with patch(
            "job_writing_agent.utils.document_loader"
            ".src.agentql_job_scraper.agentql.wrap",
            return_value=mock_page,
        ):
            mock_page.goto.return_value = None
            scraper = AgentQlJobScraper(mock_browser, strategy)
            result = scraper.scrape("https://example.com/job")

        strategy.execute.assert_called_once_with(mock_page)
        assert result.job_title == "Eng"
        assert result.method == ExtractionMethod.AQL_WITH_CONTEXT

    def test_scrape_closes_page_on_success(self) -> None:
        strategy = MagicMock()
        strategy.method_name = ExtractionMethod.AQL_WITH_CONTEXT
        strategy.execute.return_value = {}

        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page

        with patch(
            "job_writing_agent.utils.document_loader"
            ".src.agentql_job_scraper.agentql.wrap",
            return_value=mock_page,
        ):
            mock_page.goto.return_value = None
            AgentQlJobScraper(mock_browser, strategy).scrape(
                "https://example.com"
            )

        mock_page.close.assert_called_once()

    def test_scrape_closes_page_on_error(self) -> None:
        strategy = MagicMock()
        strategy.method_name = ExtractionMethod.AQL_WITH_CONTEXT
        strategy.execute.side_effect = RuntimeError("boom")

        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page

        with patch(
            "job_writing_agent.utils.document_loader"
            ".src.agentql_job_scraper.agentql.wrap",
            return_value=mock_page,
        ):
            mock_page.goto.return_value = None
            with pytest.raises(ScraperError):
                AgentQlJobScraper(mock_browser, strategy).scrape(
                    "https://example.com"
                )

        mock_page.close.assert_called_once()
