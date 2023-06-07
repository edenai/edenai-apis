from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileError: [
        r"The value you supplied as a file to process was empty or too short to process\.  Please check that you selected a valid file before submitting a processing request.",
        r"Document could not be processed",
    ]
}
