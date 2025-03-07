import importlib
import re
from enum import Enum
from typing import Dict, List, Optional, Type


class AsyncJobExceptionReason(Enum):
    DEPRECATED_JOB_ID = (
        "Either you entered a wrong job id, or you did try to get the response after the "
        "provider has already deleted your job"
    )


class ProviderException(Exception):
    """Handle error returned by providers"""

    def __init__(self, message: Optional[str] = None, code=None):
        super().__init__(message)
        if code:
            self.code = code

    @property
    def status_code(self):
        if not hasattr(self, "code"):
            return None
        return self.code


ProviderErrorLists = Dict[Type[ProviderException], List[str]]


class LanguageException(ProviderException):
    """Handle language errors"""

    def __init__(self, message: str, code=None):
        super().__init__(message, code)


class AsyncJobException(ProviderException):
    """Handle deprecated job ids"""

    def __init__(
        self,
        reason: Optional[AsyncJobExceptionReason] = None,
        message: Optional[str] = None,
        code=None,
    ):
        error_message = ""
        if message:
            error_message = message
        else:
            error_message = reason.value if reason else None
        super().__init__(error_message, code)


class ProviderInternalServerError(ProviderException):
    """Error Occuring when Provider returns an internal server error"""


class ProviderAuthorizationError(ProviderException):
    """When wrong API KEY or other auth related error"""


class ProviderPermissionError(ProviderException):
    """When the client doesn't have sufficient permissions for the request sent"""


class ProviderNotFoundError(ProviderException):
    """When resource requested is not found"""


class ProviderTimeoutError(ProviderException):
    """When provider timeout"""


class ProviderLimitationError(ProviderException):
    """Provider limit concurrent requests or other recourses"""


class ProviderParsingError(ProviderException):
    """When the provider couldn't analyze/parse inputs as expected
    Example:
        - Couldn't detect text language
        - Couldn't find face in image/video
        - Couldn't find sentiment in text
    """


class ProviderInvalidInputError(ProviderException):
    """General Error when the input sent to provider is invalid"""


class ProviderMissingInputError(ProviderException):
    """Error when a required input wasn't send"""


class ProviderInvalidInputTextLengthError(ProviderInvalidInputError):
    """When Text is Too small/long"""


class ProviderInvalidInputPayloadSize(ProviderInvalidInputError):
    """When the request body is too long"""


class ProviderInvalidInputFileError(ProviderInvalidInputError):
    """When File send is not valid"""


class ProviderInvalidInputDocumentPages(ProviderInvalidInputFileError):
    """When File send is not valid"""


class ProviderInvalidInputFileSizeError(ProviderInvalidInputFileError):
    """When File Size is too small/long"""


class ProviderInvalidInputFileFormatError(ProviderInvalidInputFileError):
    """When File has unsupported format/extension"""


class ProviderInvalidInputImageResolutionError(ProviderInvalidInputFileError):
    """When input Image Resolution is not supported"""


class ProviderInvalidInputAudioDurationError(ProviderInvalidInputFileError):
    """When input Audio is too short/long"""


class ProviderInvalidInputAudioEncodingError(ProviderInvalidInputFileError):
    """Invalid Encoding is passed"""


class ProviderInvalidPromptError(ProviderInvalidInputError):
    """When an invalid Prompt is passed to generative features"""


def get_appropriate_error(
    provider: str, exception: ProviderException
) -> ProviderException:
    """
    Given a ProviderException, check in the provider's errors list for corresponding error message
    return appropriate error if present else return original exception
    """
    try:
        provider_mod = importlib.import_module(f"apis.{provider}.errors")
    except ModuleNotFoundError:
        # we didn't implement errors yet for this provider
        return exception
    error_dict = getattr(provider_mod, "ERRORS")
    error_msg = str(exception)
    error_code = exception.status_code

    for exception_type, error_list in error_dict.items():
        if any([re.search(error_pattern, error_msg) for error_pattern in error_list]):
            return exception_type(error_msg, error_code)

    return exception
