"""Unit tests for scraper_experiment module.

All tests run without a browser or an AgentQL API key.  They cover:

- JOB_URLS list integrity
- Constant consistency (_TOTAL_CONTENT_FIELDS)
- Path-builder helpers
- _truncate helper
- ExperimentResult / ExperimentReport construction (via experiment_models)
- _serialize_report correctness and JSON round-trip
- save_json_report file I/O
- Presence of all three strategies in run_experiment source
"""

from __future__ import annotations

import inspect
import json
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
from job_writing_agent.utils.document_loader.src.scraper_experiment import (
    _TOTAL_CONTENT_FIELDS,
    _build_json_path,
    _build_log_path,
    _build_markdown_path,
    _serialize_report,
    _truncate,
    run_experiment,
    save_json_report,
    JOB_URLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_report() -> ExperimentReport:
    """An ExperimentReport with one success and one failure result."""
    extract_ok = JobExtract(
        url="https://greenhouse.io/job/1",
        method=ExtractionMethod.AQL_WITH_CONTEXT,
        job_title="Staff Engineer",
        company_name="Acme",
        populated_fields=8,
        scrape_time_ms=4500,
    )
    extract_fail = JobExtract(
        url="https://example.com/job/2",
        method=ExtractionMethod.AQL_STRUCTURED,
        has_error=True,
        error_message="Navigation failed: timeout",
    )
    return ExperimentReport(
        run_id="2026-05-01T21:40:00",
        total_trials=2,
        successful_trials=1,
        results=[
            ExperimentResult(
                url=extract_ok.url,
                method=ExtractionMethod.AQL_WITH_CONTEXT,
                extract=extract_ok,
                is_success=True,
            ),
            ExperimentResult(
                url=extract_fail.url,
                method=ExtractionMethod.AQL_STRUCTURED,
                extract=extract_fail,
                is_success=False,
                error_message="Navigation failed: timeout",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# JOB_URLS
# ---------------------------------------------------------------------------


class TestJobUrls:
    def test_exactly_10_urls(self) -> None:
        assert len(JOB_URLS) == 10

    def test_all_start_with_https(self) -> None:
        for url in JOB_URLS:
            assert url.startswith("http"), f"Non-HTTP URL: {url}"

    def test_all_unique(self) -> None:
        assert len(set(JOB_URLS)) == len(JOB_URLS)


# ---------------------------------------------------------------------------
# _TOTAL_CONTENT_FIELDS
# ---------------------------------------------------------------------------


class TestTotalContentFields:
    def test_matches_job_extract_content_fields(self) -> None:
        assert _TOTAL_CONTENT_FIELDS == len(JobExtract.CONTENT_FIELDS)

    def test_value_is_12(self) -> None:
        assert _TOTAL_CONTENT_FIELDS == 12


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_string_unchanged(self) -> None:
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self) -> None:
        assert _truncate("hello", 5) == "hello"

    def test_long_string_truncated(self) -> None:
        result = _truncate("hello world foo", 8)
        assert len(result) == 8
        assert result.endswith("…")

    def test_output_length_equals_max_len(self) -> None:
        assert len(_truncate("x" * 100, 20)) == 20


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------


class TestPathBuilders:
    def test_json_path_filename(self) -> None:
        path = _build_json_path("2026-05-01T21:40:00")
        assert path.name == "results_2026-05-01T21-40-00.json"

    def test_markdown_path_filename(self) -> None:
        path = _build_markdown_path("2026-05-01T21:40:00")
        assert path.name == "report_2026-05-01T21-40-00.md"

    def test_log_path_filename(self) -> None:
        path = _build_log_path("2026-05-01T21:40:00")
        assert path.name == "experiment_2026-05-01T21-40-00.log"

    def test_all_paths_under_results_dir(self) -> None:
        for builder in (_build_json_path, _build_markdown_path, _build_log_path):
            assert builder("2026-01-01T00:00:00").parent.name == (
                "experiment_results"
            )


# ---------------------------------------------------------------------------
# ExperimentReport / ExperimentResult (from experiment_models)
# ---------------------------------------------------------------------------


class TestExperimentReport:
    def test_construction(self, sample_report: ExperimentReport) -> None:
        assert sample_report.run_id == "2026-05-01T21:40:00"
        assert sample_report.total_trials == 2
        assert sample_report.successful_trials == 1
        assert len(sample_report.results) == 2

    def test_failed_result_has_error_message(
        self, sample_report: ExperimentReport
    ) -> None:
        failure = next(r for r in sample_report.results if not r.is_success)
        assert failure.error_message == "Navigation failed: timeout"
        assert failure.extract.has_error is True


# ---------------------------------------------------------------------------
# _serialize_report
# ---------------------------------------------------------------------------


class TestSerializeReport:
    def test_top_level_keys(self, sample_report: ExperimentReport) -> None:
        payload = _serialize_report(sample_report)
        for key in ("run_id", "total_trials", "successful_trials", "results"):
            assert key in payload

    def test_result_count(self, sample_report: ExperimentReport) -> None:
        assert len(_serialize_report(sample_report)["results"]) == 2

    def test_private_field_stripped(
        self, sample_report: ExperimentReport
    ) -> None:
        payload = _serialize_report(sample_report)
        for result in payload["results"]:
            assert "_CONTENT_FIELDS" not in result["extract"]

    def test_json_serialisable(self, sample_report: ExperimentReport) -> None:
        assert json.dumps(_serialize_report(sample_report))

    def test_json_round_trip_preserves_job_title(
        self, sample_report: ExperimentReport
    ) -> None:
        data = json.loads(json.dumps(_serialize_report(sample_report)))
        assert data["results"][0]["extract"]["job_title"] == "Staff Engineer"

    def test_failure_result_preserved(
        self, sample_report: ExperimentReport
    ) -> None:
        payload = _serialize_report(sample_report)
        failure = next(r for r in payload["results"] if not r["is_success"])
        assert failure["error_message"] == "Navigation failed: timeout"


# ---------------------------------------------------------------------------
# save_json_report
# ---------------------------------------------------------------------------


class TestSaveJsonReport:
    def test_creates_file(self, sample_report: ExperimentReport) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "results_test.json"
            save_json_report(sample_report, out)
            assert out.exists()

    def test_file_is_valid_json(self, sample_report: ExperimentReport) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.json"
            save_json_report(sample_report, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["total_trials"] == 2

    def test_creates_parent_directories(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "b" / "c" / "results.json"
            save_json_report(sample_report, out)
            assert out.exists()


# ---------------------------------------------------------------------------
# run_experiment source inspection
# ---------------------------------------------------------------------------


class TestRunExperimentSource:
    def test_all_three_strategies_present(self) -> None:
        src = inspect.getsource(run_experiment)
        assert "AqlStructuredStrategy" in src
        assert "AqlWithContextStrategy" in src
        assert "PromptExperimentalStrategy" in src

    def test_headless_true(self) -> None:
        assert "headless=True" in inspect.getsource(run_experiment)
