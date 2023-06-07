from edenai_apis.utils.exception import ProviderErrorLists, ProviderAuthorizationError

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderAuthorizationError: [
        r"Authentication required",
    ],
}
