from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"text supports maximum \d+ characters",
    ],
}
