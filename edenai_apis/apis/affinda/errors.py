from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputFileSizeError,
    ProviderInternalServerError
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputFileSizeError: [
        r"File exceeds maximum size allowed",
    ],
    ProviderInternalServerError: [
        r"^Operation returned an invalid status 'Bad Gateway' Content:.*?502 Bad Gateway.*?nginx$"
    ]
}
