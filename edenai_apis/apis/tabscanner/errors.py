from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderLimitationError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderLimitationError: [
        r"Result not yet available",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Provider tabscanner doesn't support file type: \w+ for this feature. Supported mimetypes are \w+"
    ],
}
