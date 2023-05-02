from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"invalid url or image",
    ]
}
