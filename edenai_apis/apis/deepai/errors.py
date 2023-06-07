from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"The system detected potentially unsafe content\. Please try again or adjust the prompt",
    ]
}
