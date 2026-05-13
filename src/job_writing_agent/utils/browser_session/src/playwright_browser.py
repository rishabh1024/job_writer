"""Simple, clean browser manager for AgentQL."""

import asyncio
import logging
import pprint
from typing import Optional

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
        if self._browser is None:
            await self._start()
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
        await self.create_new_context()

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
        try:
            await page.goto(url, wait_until="networkidle")

            await page.wait_for_load_state(
                state="networkidle", timeout=self.timeout_ms
            )

            # snapshot = page.accessibility
            # pprint.pprint(snapshot)

            return await page.query_data(
                query,
                wait_for_network_idle=True,
                mode="standard",
            )
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


if __name__ == "__main__":

    async def main():
        browser = AgentQLBrowser()
        page = await browser.create_agentql_page()
        query = """
                    {
                        Job_Posting_Description[]{
                            Heading[]{
                            Body (The content under the heading. Maintain the same structure as the original data. Use line separators for new paragraphs and bullets as needed.)
                            }
                            }
                    }
                """

        url = "https://altera.wd1.myworkdayjobs.com/en-US/altera/job/Bengaluru-Karnataka-India/FPGA-IP-Software-Development-Engineer_R01384-1"
        if page is not None:
            response = await browser.query_page(page, url, query)
            pprint.pprint(response)
        else:
            print("Page is not initialized")

    asyncio.run(main())
