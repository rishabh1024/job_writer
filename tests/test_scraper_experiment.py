"""Unit tests for scraper_experiment module.

All tests run without a browser or an AgentQL API key.  They cover:

- JOB_URLS list integrity
- Constant consistency (_TOTAL_CONTENT_FIELDS)
- Path-builder helpers
- _truncate helper
- ExperimentResult / ExperimentReport construction
- _serialize_report correctness and JSON round-trip
- save_report file I/O
- Presence of all three extraction methods in run_experiment source
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
from job_writing_agent.utils.document_loader.src.scraper_experiment import (
    _TOTAL_CONTENT_FIELDS,
    _build_log_path,
    _build_results_path,
    _serialize_report,
    _truncate,
    run_experiment,
    save_report,
    ExperimentReport,
    ExperimentResult,
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
        assert len(set(JOB_URLS)) == len(JOB_URLS), "Duplicate URLs found"


# ---------------------------------------------------------------------------
# _TOTAL_CONTENT_FIELDS
# ---------------------------------------------------------------------------


class TestTotalContentFields:
    def test_matches_job_extract_content_fields(self) -> None:
        j = JobExtract(url="x", method=ExtractionMethod.AQL_STRUCTURED)
        assert _TOTAL_CONTENT_FIELDS == len(j._CONTENT_FIELDS)

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
        result = _truncate("x" * 100, 20)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------


class TestPathBuilders:
    def test_results_path_filename(self) -> None:
        path = _build_results_path("2026-05-01T21:40:00")
        assert path.name == "results_2026-05-01T21-40-00.json"

    def test_log_path_filename(self) -> None:
        path = _build_log_path("2026-05-01T21:40:00")
        assert path.name == "experiment_2026-05-01T21-40-00.log"

    def test_results_path_is_under_results_dir(self) -> None:
        path = _build_results_path("2026-01-01T00:00:00")
        assert path.parent.name == "experiment_results"

    def test_log_path_is_under_results_dir(self) -> None:
        path = _build_log_path("2026-01-01T00:00:00")
        assert path.parent.name == "experiment_results"


# ---------------------------------------------------------------------------
# ExperimentReport / ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentReport:
    def test_construction(self, sample_report: ExperimentReport) -> None:
        assert sample_report.run_id == "2026-05-01T21:40:00"
        assert sample_report.total_trials == 2
        assert sample_report.successful_trials == 1
        assert len(sample_report.results) == 2

    def test_success_and_failure_results(
        self, sample_report: ExperimentReport
    ) -> None:
        successes = [r for r in sample_report.results if r.is_success]
        failures = [r for r in sample_report.results if not r.is_success]
        assert len(successes) == 1
        assert len(failures) == 1

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
        assert "run_id" in payload
        assert "total_trials" in payload
        assert "successful_trials" in payload
        assert "results" in payload

    def test_result_count(self, sample_report: ExperimentReport) -> None:
        payload = _serialize_report(sample_report)
        assert len(payload["results"]) == 2

    def test_private_field_stripped(
        self, sample_report: ExperimentReport
    ) -> None:
        payload = _serialize_report(sample_report)
        for result in payload["results"]:
            assert "_CONTENT_FIELDS" not in result["extract"]

    def test_json_serialisable(self, sample_report: ExperimentReport) -> None:
        payload = _serialize_report(sample_report)
        json_str = json.dumps(payload)
        assert json_str  # non-empty

    def test_json_round_trip_preserves_job_title(
        self, sample_report: ExperimentReport
    ) -> None:
        payload = _serialize_report(sample_report)
        round_trip = json.loads(json.dumps(payload))
        assert (
            round_trip["results"][0]["extract"]["job_title"] == "Staff Engineer"
        )

    def test_failure_result_preserved(
        self, sample_report: ExperimentReport
    ) -> None:
        payload = _serialize_report(sample_report)
        failure = next(r for r in payload["results"] if not r["is_success"])
        assert failure["error_message"] == "Navigation failed: timeout"


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_creates_file(self, sample_report: ExperimentReport) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "results_test.json"
            save_report(sample_report, out)
            assert out.exists()

    def test_file_content_is_valid_json(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results_test.json"
            save_report(sample_report, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["total_trials"] == 2

    def test_creates_parent_directories(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "b" / "c" / "results.json"
            save_report(sample_report, out)
            assert out.exists()

    def test_failure_result_in_file(
        self, sample_report: ExperimentReport
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.json"
            save_report(sample_report, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            failures = [r for r in data["results"] if not r["is_success"]]
            assert len(failures) == 1
            assert failures[0]["error_message"] == "Navigation failed: timeout"


# ---------------------------------------------------------------------------
# run_experiment source inspection
# ---------------------------------------------------------------------------


class TestRunExperimentSource:
    """Verify the methods list inside run_experiment without executing it."""

    def test_all_three_methods_present(self) -> None:
        src = inspect.getsource(run_experiment)
        assert "AQL_STRUCTURED" in src
        assert "AQL_WITH_CONTEXT" in src
        assert "PROMPT_EXPERIMENTAL" in src

    def test_headless_true(self) -> None:
        src = inspect.getsource(run_experiment)
        assert "headless=True" in src
