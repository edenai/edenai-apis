from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputFileError,
    ProviderInvalidInputImageResolutionError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Processing failed",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"Resolution is too big: \d+x\d+. Max allowed resolution: 4096x4096",
        r"Resolution is too big: exceeds limit of 178956970 pixels."
    ],
    ProviderInvalidInputFileError: [
        r"Can not load image.",
    ]

}
