import re
from pathlib import Path
from typing import Literal, cast

import pymupdf4llm
from langchain_core.documents import Document
from langgraph.types import Interrupt
from pymupdf4llm.helpers.pymupdf_rag import IdentifyHeaders

from job_writing_agent.utils.document_loader.src.file_document_loader import (
    FileDocumentLoader,
)
from job_writing_agent.utils.document_loader.src.models import (
    DocumentValidation,
)


class PDFDocumentLoader(FileDocumentLoader):
    ZW_CHARS = r"[\u200b\u200c\u200d\u200e\u200f\u2028\u2029]"
    BULLETS = r"[●•▪◦*]"

    _zw_chars = re.compile(ZW_CHARS)
    _bullets = re.compile(BULLETS)

    def __init__(self, mode: Literal["single", "page"] = "page"):
        self._mode: Literal["single", "page"] = mode

    def _check_if_file_exists(self, input_file_path: Path) -> bool:
        return input_file_path.exists() and input_file_path.is_file()

    def _is_pdf_file(self, input_file_path: Path) -> bool:
        return input_file_path.suffix.lower() == ".pdf"

    def _is_empty_pdf_file(self, input_file_path: Path) -> bool:
        if input_file_path.stat().st_size == 0:
            return True
        return False

    def validate_input_document_file(
        self, document_file: Path
    ) -> DocumentValidation:
        if not self._check_if_file_exists(document_file):
            return DocumentValidation(
                is_input_valid=False,
                error="File does not exist",
            )

        if not self._is_pdf_file(document_file):
            return DocumentValidation(
                is_input_valid=False,
                error="File is not a PDF",
            )

        if self._is_empty_pdf_file(document_file):
            return DocumentValidation(
                is_input_valid=False,
                error="File is empty",
            )

        return DocumentValidation(
            is_input_valid=True,
            error=None,
        )

    def load_document(self, document: Path) -> Document:
        pdf_headers = IdentifyHeaders(cast(str, document), body_limit=12)

        markdown_formatted_data = pymupdf4llm.to_markdown(
            doc=document, hdr_info=pdf_headers
        )

        file_metadata = {
            "file_name": document.name,
            "file_format": "pdf",
            "content_length": len(markdown_formatted_data),
        }

        if isinstance(markdown_formatted_data, list):
            markdown_formatted_data = "\n".join(
                [section["text"] for section in markdown_formatted_data]
            )

        return Document(
            page_content=markdown_formatted_data,
            metadata=file_metadata,
        )

    def validate_document(self, document: Document) -> DocumentValidation:
        return DocumentValidation(
            is_input_valid=True,
            error=None,
        )

    def generate_interrupt_to_user(self) -> str:
        user_input = Interrupt(
            value={
                "interrupt_id": "pdf_document_parse_error",
                "error_message": "File could not be parsed",
            },
            id="pdf_document_parse_error",
        )
        return cast(str, user_input)


pdf_document_loader = PDFDocumentLoader()

# result = pdf_document_loader.validate_input_document_file(
#     Path("C:/Users/risha/Downloads/Rishabh_SDE_Resume_IN.pdf")
# )

document = pdf_document_loader.load_document(
    Path("C:/Users/risha/Downloads/Rishabh_SDE_Resume_IN.pdf")
)

# print(result)
print(document)
