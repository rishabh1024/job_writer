"""Errors for the BrowserSession Class."""


class BrowserSessionError(Exception):
    def __init__(self, message, value):
        super().__init__(message)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class PlaywrightSessionTimeoutError(BrowserSessionError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class BrowserSessionAgentQLServerError(BrowserSessionError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class BrowserSessionAPIKeyError(BrowserSessionError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class BrowserSessionPageCrashError(BrowserSessionError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class BrowserSessionContextInitializationError(BrowserSessionError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}."


class BrowserSessionQuerySyntaxError(BrowserSessionError):
    def __init__(self, message, value):
        super().__init__(message, value)
        self.value = value
        self.message = message

    def __str__(self):
        return f"{self.message} {self.value}. Refer to the AgentQL documentation for the correct syntax. https://docs.agentql.com/agentql-query/best-practices"
