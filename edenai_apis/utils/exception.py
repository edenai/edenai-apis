from typing import Optional

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