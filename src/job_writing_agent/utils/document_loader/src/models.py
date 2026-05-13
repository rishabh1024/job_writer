from dataclasses import dataclass, field
from enum import StrEnum


class ErrorCode(StrEnum):
    PATH_DOES_NOT_EXIST = "FilePathDoesNotExist"
    INVALID_URL = "InvalidURL"
    INVALID_FILE_PATH = "InvalidFilePath"
    URL_NOT_REACHABLE = "URLNotReachable"
    INVALID_DOCUMENT = "InvalidDocument"
    EMPTY_DOCUMENT = "EmptyDocument"
    NONE = "None"


@dataclass
class DocumentValidation:
    is_input_valid: bool = field(default=True)
    error_code: ErrorCode | None = None
    message: str | None = None
    metadata: dict[str, str] | None = None


@dataclass
class FileMetadata:
    file_name: str
    file_format: str
    content_length: int
