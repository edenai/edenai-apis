from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputPayloadSize,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileFormatError: [
        r"File extension not supported. Use one of the following extensions: \w+"
    ],
    ProviderInvalidInputPayloadSize: [
        r"Length Required: Fetching failed to get valid content-length."
    ],
}
