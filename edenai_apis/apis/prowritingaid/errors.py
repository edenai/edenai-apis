from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputImageResolutionError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Provider does not support selected language: \w+",
        r"This provider doesn't auto-detect languages, please provide a valid language",
    ],
}
