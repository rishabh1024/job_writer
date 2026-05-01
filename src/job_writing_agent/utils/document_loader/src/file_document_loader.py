from abc import abstractmethod
from pathlib import Path

from job_writing_agent.utils.document_loader.src.document_loader import (
    DocumentLoader,
)
from job_writing_agent.utils.document_loader.src.models import (
    InputFileValidationResult,
)


class FileDocumentLoader(DocumentLoader):
    @abstractmethod
    def validate_input_document_file(
        self, document_file: Path
    ) -> InputFileValidationResult:
        """
        The input file has to be validated before it
        is passed to the file document loader.

        Check include:
        1. valid file format
        2. empty file (capture: fitz.EmptyFileError)
        3. Is the file path valid
        """
        pass
