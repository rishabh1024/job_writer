"""Simple, clean browser manager for AgentQL."""

import asyncio
import logging
import sys
import threading
from typing import Any, Coroutine, Optional, TypeVar

import agentql
from agentql import (
    AccessibilityTreeError,
    AgentQLServerError,
    APIKeyError,
    BaseAgentQLError,
    PageCrashError,
    QuerySyntaxError,
)
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from job_writing_agent.utils.browser_session.errors.errors import (
    BrowserSessionAgentQLServerError,
    BrowserSessionAPIKeyError,
    BrowserSessionContextInitializationError,
    BrowserSessionError,
    BrowserSessionPageCrashError,
    BrowserSessionQuerySyntaxError,
    PlaywrightSessionTimeoutError,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _host_loop_supports_playwright_subprocess() -> bool:
    """Playwright starts the driver with asyncio subprocess; that is not supported on every loop.

    On Windows, LangGraph / Starlette / uvicorn often run a ``WindowsSelectorEventLoop``,
    which raises ``NotImplementedError`` for subprocess transport. A
    ``ProactorEventLoop`` supports it.
    """
    if sys.platform != "win32":
        return True
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return True
    return isinstance(running_loop, asyncio.ProactorEventLoop)


class _WindowsPlaywrightDedicatedLoop:
    """Background thread holding a ``ProactorEventLoop`` for Playwright on Windows."""

    _instance_lock = threading.Lock()
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread: Optional[threading.Thread] = None
    _ready = threading.Event()

    @classmethod
    def get_loop(cls) -> asyncio.AbstractEventLoop:
        with cls._instance_lock:
            if cls._thread is None or not cls._thread.is_alive():
                cls._ready.clear()
                cls._loop = None
                cls._thread = threading.Thread(
                    target=cls._thread_main,
                    name="playwright-proactor-loop",
                    daemon=True,
                )
                cls._thread.start()
                if not cls._ready.wait(timeout=120.0):
                    raise BrowserSessionError(
                        message=(
                            "Timed out waiting for the dedicated Playwright "
                            "event loop thread (Windows)."
                        ),
                        value=None,
                    )
            if cls._loop is None:
                raise BrowserSessionError(
                    message="Playwright event loop is not available (Windows).",
                    value=None,
                )
            return cls._loop

    @classmethod
    def _thread_main(cls) -> None:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cls._loop = loop
        cls._ready.set()
        loop.run_forever()


async def _await_coroutine_on_playwright_loop(
    coroutine: Coroutine[Any, Any, _T],
) -> _T:
    """Run *coroutine* on a subprocess-capable loop and resume the caller's loop with the result."""
    if _host_loop_supports_playwright_subprocess():
        return await coroutine

    playwright_loop = _WindowsPlaywrightDedicatedLoop.get_loop()
    caller_loop = asyncio.get_running_loop()
    if playwright_loop is caller_loop:
        return await coroutine

    future = asyncio.run_coroutine_threadsafe(coroutine, playwright_loop)
    try:
        return await asyncio.wrap_future(future)
    except Exception:
        if not future.done():
            future.cancel()
        raise


class AgentQLBrowser:
    """A simple browser manager for scraping with AgentQL."""

    def __init__(
        self,
        timeout_ms: int = 30000,
        debug: bool = False,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.debug = debug
        self.headless = not debug
        self._browser: Optional[Browser] = None
        self._playwright = None
        self._context: Optional[BrowserContext] = None

    async def _start(self) -> Browser:
        """Start the browser."""
        return await _await_coroutine_on_playwright_loop(self._start_impl())

    async def _start_impl(self) -> Browser:
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            logger.info(msg="Playwright Browser is initialized.")
            return self._browser
        except PlaywrightTimeoutError as e:
            raise PlaywrightSessionTimeoutError(
                message="Failed to initialize agentql browser", value=e
            )
        except PlaywrightError as e:
            raise BrowserSessionError(
                message="Failed to initialize playwright browser", value=e
            )
        except Exception as e:
            logging.exception(f"Failed to initialize playwright browser. {e}")
            raise BrowserSessionError(
                message="Failed to initialize playwright browser", value=e
            )

    async def _close(self):
        """Close the browser and cleanup."""
        await _await_coroutine_on_playwright_loop(self._close_impl())

    async def _close_impl(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        except Exception as e:
            raise BrowserSessionError(
                message="Failed to close playwright browser", value=e
            )
        finally:
            self._browser = None

        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            raise BrowserSessionError(
                message="Failed to stop playwright", value=e
            )
        finally:
            self._playwright = None
            logger.info(msg="Playwright Browser is closed.")

    async def create_new_context(self) -> BrowserContext:
        """Create a new browser context."""
        return await _await_coroutine_on_playwright_loop(
            self._create_new_context_impl()
        )

    async def _create_new_context_impl(self) -> BrowserContext:
        if self._browser is None:
            await self._start_impl()
        if self._browser is None:
            raise BrowserSessionError(
                message="Browser is not initialized after startup", value=None
            )
        if self._context is None:
            try:
                self._context = await self._browser.new_context()
            except Exception as e:
                logging.exception("Failed to create browser context: %s", e)
                raise BrowserSessionContextInitializationError(
                    message="Failed to create new browser context",
                    value=e,
                ) from e
        return self._context

    async def create_agentql_page(self):
        """Open a new tab in the shared context and wrap it with AgentQL."""
        return await _await_coroutine_on_playwright_loop(
            self._create_agentql_page_impl()
        )

    async def _create_agentql_page_impl(self):
        await self._create_new_context_impl()

        try:
            if self._context is None:
                raise BrowserSessionError(
                    message="Context is not initialized", value=None
                )
            page = await self._context.new_page()
            page.set_default_timeout(self.timeout_ms)
            return await agentql.wrap_async(page)
        except AccessibilityTreeError as e:
            raise BrowserSessionError(
                message="Failed to create new page", value=e
            )
        except BaseAgentQLError as e:
            raise BrowserSessionError(
                message="Failed to create new page", value=e
            )
        except Exception as e:
            logging.exception(f"Failed to create new page. Error: {e}")

    async def query_page(self, page: Page, url: str, query: str) -> dict:
        """Query the page with the given query."""
        return await _await_coroutine_on_playwright_loop(
            self._query_page_impl(page, url, query)
        )

    async def _query_page_impl(self, page: Page, url: str, query: str) -> dict:
        try:
            await page.goto(url, wait_until="networkidle")

            await page.wait_for_load_state(
                state="networkidle", timeout=self.timeout_ms
            )

            page_query_response = await page.query_data(
                query,
                wait_for_network_idle=True,
                mode="standard",
            )
            return page_query_response
        except APIKeyError as e:
            raise BrowserSessionAPIKeyError(
                message="Failed to query page", value=e
            )
        except QuerySyntaxError as e:
            raise BrowserSessionQuerySyntaxError(
                message=f"Failed to query page. Query : {query}", value=e
            )
        except PageCrashError:
            raise BrowserSessionPageCrashError(
                message="Failed to query page", value=page
            )
        except (PlaywrightTimeoutError, TimeoutError) as e:
            raise PlaywrightSessionTimeoutError(
                message="Failed to query page", value=e
            )
        except AgentQLServerError as e:
            raise BrowserSessionAgentQLServerError(
                message=f"Failed to query page. AgentQL Server Error: {e}",
                value=e,
            )
        except Exception as e:
            logging.exception(f"Failed to query page. Error: {e}")
            return {"error": str(e)}

    async def __aenter__(self) -> "AgentQLBrowser":
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc_val, traceback) -> None:
        await self._close()
        logger.info(msg="Playwright Browser Session is closed.")


# if __name__ == "__main__":

#     async def main():
#         browser = AgentQLBrowser()
#         page = await browser.create_agentql_page()
#         query = """
#                     {
#                         Job_Posting_Description[]{
#                             Heading[]{
#                             Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
#                             }
#                             }
#                     }
#                 """

#         url = "https://altera.wd1.myworkdayjobs.com/en-US/altera/job/Bengaluru-Karnataka-India/FPGA-IP-Software-Development-Engineer_R01384-1"
#         if page is not None:
#             response = await browser.query_page(page, url, query)
#             pprint.pprint(response)
#         else:
#             print("Page is not initialized")

#     asyncio.run(main())
