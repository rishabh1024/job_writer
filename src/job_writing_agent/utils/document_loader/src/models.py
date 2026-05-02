from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_INPUT = "InvalidInput"
    URL_NOT_REACHABLE = "URLNotReachable"
    EMPTY_DOC = "EmptyDocument"
    INVALID_URL = "InvalidURL"
    NONE = "None"


@dataclass
class DocumentValidation:
    is_input_valid: bool = field(default=True)
    error_code: ErrorCode | None = None
    message: str | None = None


@dataclass
class FileMetadata:
    file_name: str
    file_format: str
    content_length: int
