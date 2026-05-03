"""Run every AQL strategy against every URL and stream results to a markdown report."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import agentql
from playwright.sync_api import Browser, sync_playwright

from job_writing_agent.utils.document_loader.src.queries import STRATEGIES

RESULTS_DIR: Path = Path(__file__).parent / "experiment_results"
PAGE_NAV_TIMEOUT_MS: int = 30_000
QUERY_TIMEOUT_S: int = 10

JOB_URLS: list[str] = [
    "https://autodesk.wd1.myworkdayjobs.com/en-US/Ext/job/Pune%2C-IND/Senior-Software-Engineer_25WD93636-1?src=JB-10065&source=LinkedIn",
    "https://paypal.wd1.myworkdayjobs.com/en-US/jobs/job/Bangalore-Karnataka-India/Senior-Software-Engineer---Backend--Java-_R0134858",
    "https://fox.wd1.myworkdayjobs.com/en-US/Domestic/job/IND-KA-Bengaluru/Senior-Software-Development-Engineer--Backend_R50031537",
    "https://altera.wd1.myworkdayjobs.com/en-US/altera/job/Bengaluru-Karnataka-India/FPGA-IP-Software-Development-Engineer_R01384-1",
]


def _timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%S")


def _write_success(md: TextIO, name: str, data: dict, elapsed_ms: int) -> None:
    md.write(f"### Strategy: {name}\n\n")
    md.write(f"- Time: {elapsed_ms} ms\n\n")
    md.write("```json\n")
    md.write(json.dumps(data, indent=2, ensure_ascii=False))
    md.write("\n```\n\n")


def _write_error(
    md: TextIO, name: str, exc: Exception, elapsed_ms: int
) -> None:
    md.write(f"### Strategy: {name}\n\n")
    md.write(f"- Time: {elapsed_ms} ms\n")
    md.write(f"- Error: `{exc}`\n\n")


def _run_trial(
    browser: Browser, url: str, name: str, query: str, md: TextIO
) -> None:
    page = agentql.wrap(browser.new_page())
    start = time.perf_counter()
    try:
        page.goto(
            url, timeout=PAGE_NAV_TIMEOUT_MS, wait_until="domcontentloaded"
        )
        page.wait_for_page_ready_state()
        data = page.query_data(
            query, timeout=QUERY_TIMEOUT_S, ResponseMode="standard"
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        _write_success(md, name, data, elapsed_ms)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        _write_error(md, name, exc, elapsed_ms)
    finally:
        page.close()
        md.flush()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"report_{_timestamp()}.md"

    with (
        sync_playwright() as playwright,
        playwright.chromium.launch(headless=True) as browser,
        report_path.open("w", encoding="utf-8") as markdown_file,
    ):
        markdown_file.write(
            f"# AgentQL Job Scraper Report — {_timestamp()}\n\n"
        )
        markdown_file.flush()
        for url in JOB_URLS:
            markdown_file.write(f"## {url}\n\n")
            markdown_file.flush()
            for name, query in STRATEGIES.items():
                print(f"  {url}  ->  {name}")
                _run_trial(browser, url, name, query, markdown_file)


if __name__ == "__main__":
    main()
