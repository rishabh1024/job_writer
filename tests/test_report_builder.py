"""Unit tests for results/report_builder.py.

All tests run without a browser or API key.  They cover:

- _render_field_value: None, empty list, list with items, plain string
- _render_field_table: column count, populated vs missing fields
- build_markdown_report: structure, URL headings, strategy headings,
  summary table presence, full content
- save_markdown_report: file creation, valid UTF-8, parent directory creation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from job_writing_agent.utils.document_loader.src.agentql_job_scraper import (
    ExtractionMethod,
    JobExtract,
)
from job_writing_agent.utils.document_loader.src.experiment_models import (
    ExperimentReport,
    ExperimentResult,
)
from job_writing_agent.utils.document_loader.src.results.report_builder import (
    _render_field_value,
    _render_field_table,
    build_markdown_report,
    save_markdown_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_extract() -> JobExtract:
    """A fully-populated ``JobExtract``."""
    return JobExtract(
        url="https://greenhouse.io/job/1",
        method=ExtractionMethod.AQL_WITH_CONTEXT,
        job_title="Staff Engineer",
        company_name="Acme Corp",
        job_location="London, UK",
        employment_type="Full-time",
        salary_range="£80k–£110k",
        job_summary="Build scalable systems.",
        responsibilities=["Design APIs", "Mentor juniors"],
        requirements=["5 years Python"],
        preferred_qualifications=["PhD"],
        benefits=["Health", "Pension"],
        application_deadline="2026-12-01",
        remote_policy="Hybrid",
        populated_fields=12,
        scrape_time_ms=3_200,
    )


@pytest.fixture()
def failed_extract() -> JobExtract:
    """A ``JobExtract`` that represents a scrape failure."""
    return JobExtract(
        url="https://lever.co/job/99",
        method=ExtractionMethod.AQL_STRUCTURED,
        has_error=True,
        error_message="Navigation failed: timeout",
    )


@pytest.fixture()
def sample_report(
    full_extract: JobExtract, failed_extract: JobExtract
) -> ExperimentReport:
    """A minimal ``ExperimentReport`` with two results."""
    return ExperimentReport(
        run_id="2026-05-02T10:00:00",
        total_trials=2,
        successful_trials=1,
        results=[
            ExperimentResult(
                url=full_extract.url,
                method=ExtractionMethod.AQL_WITH_CONTEXT,
                extract=full_extract,
                is_success=True,
            ),
            ExperimentResult(
                url=failed_extract.url,
                method=ExtractionMethod.AQL_STRUCTURED,
                extract=failed_extract,
                is_success=False,
                error_message="Navigation failed: timeout",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# _render_field_value
# ---------------------------------------------------------------------------


class TestRenderFieldValue:
    def test_none_renders_not_found(self) -> None:
        assert _render_field_value(None) == "_not found_"

    def test_empty_string_renders_not_found(self) -> None:
        assert _render_field_value("") == "_not found_"

    def test_empty_list_renders_not_found(self) -> None:
        assert _render_field_value([]) == "_not found_"

    def test_plain_string_returned_as_is(self) -> None:
        assert _render_field_value("Full-time") == "Full-time"

    def test_list_renders_bullet_items(self) -> None:
        result = _render_field_value(["Design APIs", "Mentor juniors"])
        assert "• Design APIs" in result
        assert "• Mentor juniors" in result

    def test_list_uses_br_separator(self) -> None:
        result = _render_field_value(["A", "B"])
        assert "<br>" in result

    def test_list_with_only_empty_strings_renders_not_found(self) -> None:
        assert _render_field_value(["", " "]) == "_not found_"


# ---------------------------------------------------------------------------
# _render_field_table
# ---------------------------------------------------------------------------


class TestRenderFieldTable:
    def test_contains_header_row(self, full_extract: JobExtract) -> None:
        table = _render_field_table(full_extract)
        assert "| Field | Value |" in table

    def test_contains_separator_row(self, full_extract: JobExtract) -> None:
        table = _render_field_table(full_extract)
        assert "|---|---|" in table

    def test_all_12_fields_present(self, full_extract: JobExtract) -> None:
        table = _render_field_table(full_extract)
        for label in [
            "Job Title",
            "Company",
            "Location",
            "Employment Type",
            "Salary Range",
            "Remote Policy",
            "Application Deadline",
            "Job Summary",
            "Responsibilities",
            "Requirements",
            "Preferred Qualifications",
            "Benefits",
        ]:
            assert label in table, f"Missing field label: {label}"

    def test_populated_value_appears(self, full_extract: JobExtract) -> None:
        table = _render_field_table(full_extract)
        assert "Staff Engineer" in table
        assert "Acme Corp" in table

    def test_missing_value_shows_not_found(self) -> None:
        sparse = JobExtract(
            url="x", method=ExtractionMethod.AQL_STRUCTURED
        )
        table = _render_field_table(sparse)
        assert "_not found_" in table


# ---------------------------------------------------------------------------
# build_markdown_report
# ---------------------------------------------------------------------------


class TestBuildMarkdownReport:
    def test_contains_title(self, sample_report: ExperimentReport) -> None:
        md = build_markdown_report(sample_report)
        assert "# AgentQL Job Description Scraper" in md

    def test_contains_run_id(self, sample_report: ExperimentReport) -> None:
        md = build_markdown_report(sample_report)
        assert "2026-05-02T10:00:00" in md

    def test_contains_summary_heading(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "## Summary" in md

    def test_summary_table_has_strategy_rows(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "Aql With Context" in md or "aql_with_context" in md

    def test_url_appears_as_heading(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "https://greenhouse.io/job/1" in md
        assert "https://lever.co/job/99" in md

    def test_strategy_headings_present(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "### Strategy:" in md

    def test_failed_result_shows_error(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "FAILED" in md
        assert "Navigation failed: timeout" in md

    def test_successful_result_shows_fields_count(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "12 / 12" in md

    def test_successful_result_shows_scrape_time(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "3,200 ms" in md

    def test_full_text_fields_not_truncated(
        self, sample_report: ExperimentReport
    ) -> None:
        md = build_markdown_report(sample_report)
        assert "Build scalable systems." in md
        assert "Design APIs" in md
        assert "Mentor juniors" in md

    def test_returns_string(self, sample_report: ExperimentReport) -> None:
        assert isinstance(build_markdown_report(sample_report), str)


# ---------------------------------------------------------------------------
# save_markdown_report
# ---------------------------------------------------------------------------


class TestSaveMarkdownReport:
    def test_creates_file(self, sample_report: ExperimentReport) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            save_markdown_report(sample_report, out)
            assert out.exists()

    def test_file_is_valid_utf8(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            save_markdown_report(sample_report, out)
            content = out.read_text(encoding="utf-8")
            assert "AgentQL" in content

    def test_creates_parent_directories(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "b" / "report.md"
            save_markdown_report(sample_report, out)
            assert out.exists()

    def test_content_matches_build_output(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            save_markdown_report(sample_report, out)
            assert out.read_text(encoding="utf-8") == build_markdown_report(
                sample_report
            )
