from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Error calling Clarifai API",
        r"Failure",
    ],
}
