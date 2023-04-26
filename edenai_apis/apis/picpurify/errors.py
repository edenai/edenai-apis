from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputImageResolutionError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Unable to process this image",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"Image size too large : \d+x\d+, Maximum size authorized: 4096x4096",
    ],
}
