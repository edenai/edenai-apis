from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileSizeError,
    ProviderParsingError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderParsingError: [
        r"Provider did not return any face",
    ],
    ProviderInvalidInputFileSizeError: [
        r"max image size is more than 2\.0 MB",
    ],
}
