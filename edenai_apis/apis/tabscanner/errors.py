from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderLimitationError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderLimitationError: [
        r"Result not yet available",
    ]
}
