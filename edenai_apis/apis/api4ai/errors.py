from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputImageResolutionError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Processing failed",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"Resolution is too big: \d+x\d+. Max allowed resolution: 4096x4096",
    ]
}
