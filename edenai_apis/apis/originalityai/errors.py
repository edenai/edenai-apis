from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"content must be at least 100 words",
    ]
}
