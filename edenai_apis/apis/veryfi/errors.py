from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputImageResolutionError,
    ProviderInvalidInputFileFormatError
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputImageResolutionError: [
        r"File is too small. Minimal size is 100x100",
    ],
    ProviderInvalidInputFileFormatError : [
        r"The file type is not enabled for your client"
    ]
}

