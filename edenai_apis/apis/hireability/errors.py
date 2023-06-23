from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileError,
    ProviderInvalidInputFileFormatError
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileError: [
        r"The value you supplied as a file to process was empty or too short to process\.  Please check that you selected a valid file before submitting a processing request.",
        r"Document could not be processed",
    ],
    ProviderInvalidInputFileFormatError : [
        r"Provider hireability doesn't support file type: \w+ for this feature. Supported mimetypes are \w+"
    ]
}
