from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"The text must not be greater than 4000 characters.",
    ],
    ProviderInvalidInputError: [
        r"The language defined in output locale is not yet supported",
    ]

}
