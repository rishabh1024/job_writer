from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from langchain_core.documents import Document

from job_writing_agent.utils.document_loader.src.models import (
    DocumentValidation,
)

T = TypeVar("T")


class DocumentLoader(ABC, Generic[T]):
    @abstractmethod
    def _validate_document_source(
        self, input_document: T
    ) -> DocumentValidation:
        """
        Validate the source of the document before assigning it to the document loader.

        Check whether the input is a valid, accessible document source. The concrete
        implmentations must define what constitutes a valid document source for
        their respective loader.

        Args:
        input_document (String, HttpURL, Path) : Path to the resource

        Returns:
        DocumentValidation:
            - is_input_valid (bool): True if the source is valid.
            - error_code (ErrorCode | None): Error code if invalid, else None.
            - message (str | None): Human-readable error detail, else None.

        Raises:
            NotImplementedError: If the subclasses do not implement this method
        """
        pass

    @abstractmethod
    def load_document(self, input_document: T) -> Document:
        """
        This method fetches the data from the input source using a web-scraper or
        file-reader and returns the user with context ready data for the agent.

        Args:
            - input_document (String, HttpURL, Path) : Location of the document
        Returns:
            - Document: a Langchain Document
        Raises:
            - NotImplementedError: If the subclasses do not implement this method
        """
        pass

    @abstractmethod
    def _validate_output_document(
        self, input_document: dict[str, list[dict[str, str]]]
    ) -> DocumentValidation:
        """
        The content of the document has to be checked for consistency and correctness.
        Inconsistent data will lead to unacceptable results.

        Args:
            - document (Document): The document to be validated
        Returns:
            - DocumentValidation:
                - is_input_valid (bool): True if the content is valid.
                - error_code (ErrorCode | None): Error code if invalid, else None.
                - message (str | None): Human-readable error detail, else None.
        Raises:
            - NotImplementedError: If the subclasses do not implement this method
        """
        pass

    @abstractmethod
    def generate_interrupt_to_user(self) -> str:
        """
        If the agent is not able to load the document after `n` retries, it will generate an interrupt to the user and prompt the user to upload the correct document path or URL.

        """
        pass
