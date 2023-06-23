from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Invalid language format for: \w+.",
        r"Language pair not supported: \w+ > \w+"
    ],
}
