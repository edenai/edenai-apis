from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderAuthorizationError,
    ProviderTimeoutError,
    ProviderInvalidInputImageResolutionError,
    ProviderParsingError,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderAuthorizationError: [
        r"Authentication required",
    ],
    ProviderTimeoutError: [
        r"\('Connection broken: ConnectionResetError\(104, 'Connection reset by peer'\)', ConnectionResetError\(104, 'Connection reset by peer'\)\)"
    ],
    ProviderInvalidInputImageResolutionError: [
        r"Image dimensions are \d+x\d+ but should be above 50x50"
    ],
    ProviderParsingError: [r"Unable to scan the document"],
    ProviderInvalidInputError: [
        r"No face was found in the \w+ photo",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Provider \w+ doesn't support file type: \w+ for this feature. Supported mimetypes are \w+"
    ],
}
