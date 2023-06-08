from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileFormatError: [
        r"File type not managed",
    ],
    ProviderInternalServerError: [
        r"500 Internal Server Error",
    ]
}
