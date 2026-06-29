import json
import logging

from langchain_core.documents import Document
from playwright.async_api import Page
from pydantic import HttpUrl, TypeAdapter, ValidationError

from job_writing_agent.utils.browser_session.errors.errors import (
    BrowserSessionAgentQLServerError,
    BrowserSessionAPIKeyError,
    BrowserSessionContextInitializationError,
    BrowserSessionPageCrashError,
    BrowserSessionQuerySyntaxError,
    PlaywrightSessionTimeoutError,
)
from job_writing_agent.utils.browser_session.src.playwright_browser import (
    AgentQLBrowser,
)
from job_writing_agent.utils.document_loader.agentql_queries import (
    DEFAULT_AQL_QUERY,
    GREENHOUSEJOB_AQL_QUERY,
    LEVER_AQL_QUERY,
    WORKDAYJOB_AQL_QUERY,
)
from job_writing_agent.utils.document_loader.errors.errors import (
    DocumentLoaderContextInitializationError,
    DocumentLoaderError,
    DocumentLoaderPageInitializationError,
    WebDocumentInvalidSourceError,
    WebDocumentOutputValidationError,
)
from job_writing_agent.utils.document_loader.src.document_loader import (
    DocumentLoader,
)
from job_writing_agent.utils.document_loader.src.models import (
    DocumentValidation,
    ErrorCode,
)

logger = logging.getLogger(__name__)


class WebDocumentLoader(DocumentLoader):
    def __init__(self, browser_session: AgentQLBrowser):
        self.browser_session = browser_session

    async def _get_agentql_page(self) -> Page | None:
        try:
            document_page = await self.browser_session.create_agentql_page()
            return document_page
        except BrowserSessionContextInitializationError as e:
            raise DocumentLoaderContextInitializationError(
                message="Failed to create new context in WebDocumentLoader",
                value=e,
            ) from e
        except BrowserSessionPageCrashError as e:
            raise DocumentLoaderPageInitializationError(
                message="Failed to create new page in WebDocumentLoader",
                value=e,
            ) from e
        except Exception as e:
            raise DocumentLoaderError(
                message="DocumentLoaderError.",
                value=e,
            ) from e

    def _url_parse(self, input_url: HttpUrl | str) -> HttpUrl:
        return TypeAdapter(HttpUrl).validate_python(input_url)

    def _get_agentql_query(self, document_url: HttpUrl):
        host_name = document_url.host
        if host_name and "myworkdayjobs" in host_name:
            return WORKDAYJOB_AQL_QUERY
        elif host_name and "lever.co" in host_name:
            return LEVER_AQL_QUERY
        elif host_name and "greenhouse.io" in host_name:
            return GREENHOUSEJOB_AQL_QUERY
        else:
            return DEFAULT_AQL_QUERY

    def _validate_document_source(
        self, document_url: HttpUrl | str
    ) -> DocumentValidation:
        """
        1. Check if the format of the url is valid
        """
        try:
            self._url_parse(document_url)
        except ValidationError:
            return DocumentValidation(
                is_input_valid=False,
                error_code=ErrorCode.INVALID_URL,
            )
        return DocumentValidation(
            is_input_valid=True, error_code=ErrorCode.NONE
        )

    async def load_document(self, document_url: HttpUrl) -> Document:
        validation = self._validate_document_source(document_url)
        if not validation.is_input_valid:
            raise WebDocumentInvalidSourceError(
                message=validation.message,
                value=validation.error_code,
            )

        agentql_query = self._get_agentql_query(document_url)
        try:
            agentql_wrapped_page = await self._get_agentql_page()
            if agentql_wrapped_page is None:
                raise DocumentLoaderPageInitializationError(
                    message="Failed to create AgentQL page",
                    value=None,
                )
            query_response = await self.browser_session.query_page(
                page=agentql_wrapped_page,
                url=str(document_url),
                query=agentql_query,
            )
            validation = self._validate_output_document(query_response)
            if not validation.is_input_valid:
                raise WebDocumentOutputValidationError(
                    message=validation.message,
                    value=validation.error_code,
                )
            company_name = query_response.get("Company_Name", "") or ""
            document_metadata = dict(validation.metadata or {})
            document_metadata["company_name"] = company_name
            if not company_name:
                raise WebDocumentOutputValidationError(
                    message="Document is invalid. Company name is empty",
                    value=ErrorCode.INVALID_DOCUMENT,
                )
            query_response_string = json.dumps(query_response, indent=2)
            return Document(
                page_content=query_response_string,
                metadata=document_metadata,
            )
        except DocumentLoaderPageInitializationError as e:
            raise DocumentLoaderPageInitializationError(
                message=e.message,
                value=e.value,
            ) from e
        except WebDocumentOutputValidationError as e:
            raise e
        except DocumentLoaderError as e:
            raise e
        except BrowserSessionAPIKeyError as e:
            raise BrowserSessionAPIKeyError(
                message="Document Load Failed. API Key Error", value=e
            ) from e
        except BrowserSessionQuerySyntaxError as e:
            raise BrowserSessionQuerySyntaxError(
                message="Document Load Failed. Query Syntax Error", value=e
            ) from e
        except PlaywrightSessionTimeoutError as e:
            raise PlaywrightSessionTimeoutError(
                message="Document Load Failed. Playwright Session Timeout Error",
                value=e,
            ) from e
        except BrowserSessionAgentQLServerError as e:
            raise BrowserSessionAgentQLServerError(
                message="Document Load Failed. AgentQL Server Error", value=e
            ) from e
        except Exception as e:
            logger.exception(
                msg=f"Unknown error in WebDocumentLoader. Exception: {e}"
            )
            raise DocumentLoaderError(
                message="Document Load Failed. Unknown Error", value=e
            ) from e

    def generate_interrupt_to_user(self) -> str:
        """
        If the agent is not able to load the document after `n` retries, it will generate an interrupt to the user and prompt the user to upload the correct document path or URL.
        """
        return "Please upload the correct document path or URL."

    def _validate_output_document(
        self, input_document: dict[str, list[dict[str, str]]]
    ) -> DocumentValidation:
        document_metadata: dict[str, str] = {}

        input_document_data = input_document.get(
            "Job_Posting_Description", None
        )

        # First validation is check if the dict has no keys or values
        if not input_document_data:
            return DocumentValidation(
                is_input_valid=False,
                error_code=ErrorCode.INVALID_DOCUMENT,
                message="Document is invalid. Job Description is empty",
            )

        # Validate if the keys have non-empty body
        for section in input_document_data:
            heading, body = (
                section.get("Heading", None),
                section.get("Body", None),
            )
            if not body or not heading:
                return DocumentValidation(
                    is_input_valid=False,
                    error_code=ErrorCode.INVALID_DOCUMENT,
                    message="Document is invalid. Heading or body is empty",
                )
            document_metadata[heading] = body
        return DocumentValidation(
            is_input_valid=True,
            error_code=ErrorCode.NONE,
            metadata=document_metadata,
            message="ValidDocument",
        )
