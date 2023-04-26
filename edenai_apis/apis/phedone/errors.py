from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"The text must not be greater than 4000 characters.",
    ]
}
