from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from langchain_core.documents import Document

from job_writing_agent.utils.document_loader.src.models import (
    DocumentValidation,
)

T = TypeVar("T")


class DocumentLoader(ABC, Generic[T]):
    @abstractmethod
    def load_document(self, document: T) -> Document:
        pass

    @abstractmethod
    def validate_document(self, document: Document) -> DocumentValidation:
        pass

    @abstractmethod
    def generate_interrupt_to_user(self) -> str:
        """
        If the agent is not able to load the document after `n` retries, it will generate an interrupt to the user and prompt the user to upload the correct document path or URL.

        """
        pass

    def retry(self, fn: Callable[[Any], Any]) -> Any:
        def wrapper(fn: Callable[[Any], Any]) -> Any:
            pass

        return wrapper
