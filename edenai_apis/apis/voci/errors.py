from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderAuthorizationError,
    ProviderInvalidInputPayloadSize
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderAuthorizationError: [
        r"Response Status: 401. Response Content: b'Invalid token",
    ],
    ProviderInvalidInputPayloadSize : [
        r"Request Entity Too Large"
    ]
}
