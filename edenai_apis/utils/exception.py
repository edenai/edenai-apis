from typing import Optional
from enum import Enum

class AsyncJobExceptionReason(Enum):
    DEPRECATED_JOB_ID = ("Either you entered a wrong job id, or you did try to get the response after the "
                          "provider has already deleted your job")

class ProviderException(Exception):
    """Handle error returned by providers"""

    def __init__(self, message: Optional[str] = None, code=None):
        super().__init__(message)
        if code:
            self.code = code


class LanguageException(ProviderException):
    """Handle language errors"""

    def __init__(self, message: str, code=None):
        super().__init__(message, code)


class AsyncJobException(ProviderException):
    """Handle deprecated job ids"""

    def __init__(self, reason: Optional[AsyncJobExceptionReason] = None, message: Optional[str] = None, code=None):
        error_message = ""
        if message:
            error_message = message
        else:
            error_message = reason.value
        super().__init__(error_message, code)