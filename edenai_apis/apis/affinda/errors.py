from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileSizeError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileSizeError: [
        r"File exceeds maximum size allowed",
    ]
}
