from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderMissingInputError,
    ProviderParsingError,
    ProviderInternalServerError
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderMissingInputError: [
        r"Detected non-English language, use multilingual flag to enable",
    ],
    ProviderParsingError: [
        r"Failed to parse input",
        r"HTML input requires an HTML-to-text preprocessing skill"
    ],
    ProviderInternalServerError : [
        r"Server error"
    ]
}
