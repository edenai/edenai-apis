from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderMissingInputError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderMissingInputError: [
        r"Missing target_lang",
    ],
    ProviderInvalidInputError: [
        r"Value for 'target_lang' not supported",
    ]
}
