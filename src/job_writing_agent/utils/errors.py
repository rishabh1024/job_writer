class ModelNotFoundError(Exception):
    """Exception raised when a requested model is not found."""
    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found.")
        self.model_name = model_name
    
    def __str__(self):
        return f"ModelNotFoundError: {self.model_name}"

class URLExtractionError(Exception):
    """Raised when content cannot be extracted from a URL."""
    pass

class LLMProcessingError(Exception):
    """Raised when LLM processing fails."""
    pass

class JobDescriptionParsingError(Exception):
    """Base class for job description parsing errors."""
    pass


class ResumeDownloadError(Exception):
    """Raised when a resume file cannot be downloaded from a URL."""
    pass