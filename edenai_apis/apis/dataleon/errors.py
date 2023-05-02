from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileFormatError: [
        r"File type not managed",
    ]
}
