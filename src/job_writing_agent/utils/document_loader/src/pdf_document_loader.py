import re
from pathlib import Path
from typing import Literal, cast

import pymupdf4llm
from langchain_core.documents import Document
from langgraph.types import Interrupt
from pymupdf4llm.helpers.pymupdf_rag import IdentifyHeaders

from job_writing_agent.utils.document_loader.errors.errors import (
    PDFDocumentValidationError,
)
from job_writing_agent.utils.document_loader.src.file_document_loader import (
    FileDocumentLoader,
)
from job_writing_agent.utils.document_loader.src.models import (
    DocumentValidation,
    ErrorCode,
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

    def _validate_document_source(
        self, document_file: Path
    ) -> DocumentValidation:
        if not self._check_if_file_exists(document_file):
            return DocumentValidation(
                is_input_valid=False,
                error_code=ErrorCode.PATH_DOES_NOT_EXIST,
                message="File does not exist",
            )

        if not self._is_pdf_file(document_file):
            return DocumentValidation(
                is_input_valid=False,
                error_code=ErrorCode.INVALID_DOCUMENT,
                message="File is not a PDF",
            )

        if self._is_empty_pdf_file(document_file):
            return DocumentValidation(
                is_input_valid=False,
                error_code=ErrorCode.EMPTY_DOCUMENT,
                message="File is empty",
            )

        return DocumentValidation(
            is_input_valid=True,
            error_code=ErrorCode.NONE,
            message=None,
        )

    def load_document(self, document: Path) -> Document:
        document_validation = self._validate_document_source(document)

        if not document_validation.is_input_valid:
            raise PDFDocumentValidationError(
                f"Document is not valid. Document: {document}.",
                document_validation.error_code,
            )

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

    def _validate_output_document(
        self, document: Document
    ) -> DocumentValidation:
        return DocumentValidation(
            is_input_valid=True,
            error_code=ErrorCode.NONE,
            message=None,
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
