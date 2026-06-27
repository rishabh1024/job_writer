class DocumentLoaderError(Exception):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class WebDocumentInvalidSourceError(DocumentLoaderError):
    pass


class WebDocumentOutputValidationError(DocumentLoaderError):
    pass


class DocumentLoaderContextInitializationError(DocumentLoaderError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class DocumentLoaderPageInitializationError(DocumentLoaderError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class PDFDocumentValidationError(DocumentLoaderError):
    pass
