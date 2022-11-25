from typing import Optional

class ProviderException(Exception):
    """Handle error returned by providers"""

    def __init__(self, message: Optional[str] = None, code=None):
        super().__init__(message)
        if code:
            self.code = code
