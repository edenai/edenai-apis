from typing import Optional


# Just a wrapper for errors
class ModelNotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LLMEngineError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ProviderAPIError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CompletionClientError(Exception):
    def __init__(self, message, input: Optional[dict] = None, status_code: int = 400):
        self.input = input
        self.message = message
        self.status_code = status_code
        super().__init__(
            self.message
        )  # TODO polymorph this thing to understand different exceptions and handle them on the logging...