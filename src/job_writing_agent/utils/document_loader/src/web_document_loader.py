from pprint import pprint
from typing import cast

from langchain_community.document_loaders import (
    AsyncChromiumLoader,
)
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from pydantic import HttpUrl, TypeAdapter, ValidationError

from job_writing_agent.utils.document_loader.src.document_loader import (
    DocumentLoader,
)
from job_writing_agent.utils.document_loader.src.models import (
    DocumentValidation,
    ErrorCode,
)


class WebDocumentLoader(DocumentLoader):
    def __init__(self):
        self.html_to_text_transformer = Html2TextTransformer()

    def _url_parse(self, input_url: HttpUrl | str) -> HttpUrl:
        return TypeAdapter(HttpUrl).validate_python(input_url)

    def _get_document_loader(self, document_url: HttpUrl):
        host_name = document_url.host

        if host_name is None:
            raise ValueError("Invalid URL")
        if "workday" not in host_name.lower():
            return AsyncChromiumLoader([cast(str, document_url)], headless=True)
        else:
            return AsyncChromiumLoader([cast(str, document_url)], headless=True)

    def validate_input_document_url(
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

    def load_document(self, document_url: HttpUrl) -> Document:
        loader = self._get_document_loader(document_url)
        try:
            web_scraped_data_ = loader.load()
            aggregated_web_scraped_data = ""

            print(web_scraped_data_)
            if web_scraped_data_:
                markdown_web_scraped_data = (
                    self.html_to_text_transformer.transform_documents(
                        web_scraped_data_
                    )
                )

                if markdown_web_scraped_data:
                    for document in markdown_web_scraped_data:
                        aggregated_web_scraped_data += document.page_content
                else:
                    raise ValueError("Failed to Scrape Data from the URL")

            return Document(
                page_content=aggregated_web_scraped_data,
                metadata={"source": document_url},
            )
        except Exception as e:
            raise ValueError(f"Failed to Scrape Data from the URL: {e}") from e

    def validate_document(self, document: Document) -> DocumentValidation:
        """
        1. Check if the document is valid
        """
        if not document.page_content:
            return DocumentValidation(
                is_input_valid=False,
                error_code=ErrorCode.EMPTY_DOC,
            )
        return DocumentValidation(
            is_input_valid=True, error_code=ErrorCode.NONE
        )

    def generate_interrupt_to_user(self) -> str:
        """
        If the agent is not able to load the document after `n` retries, it will generate an interrupt to the user and prompt the user to upload the correct document path or URL.
        """
        return "Please upload the correct document path or URL."


web_document_loader = WebDocumentLoader()
document_url = "https://proofpoint.wd5.myworkdayjobs.com/ProofpointCareers/job/Bengaluru-India---Remote/Senior-Software-Engineer_R13403?source=LinkedIn"

pprint(
    web_document_loader.parse_bot_loader(
        TypeAdapter(HttpUrl).validate_python(document_url)
    )
)
