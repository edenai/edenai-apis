from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderMissingInputError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderMissingInputError: [
        r"Missing target_lang",
    ],
    ProviderInvalidInputError: [
        r"Value for \w+ not supported",
        r"Source and target language are equal.",
    ],
    ProviderInvalidInputFileSizeError: [r"Document exceeds the size limit of 10 MB."],
    ProviderInvalidInputFileFormatError: [
        r"Failed to get the document_type from the file extension \(file\)."
        r"Provider deepl doesn't support file type: \w+ for this feature. Supported mimetypes are \w+"
    ],
}
