from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputImageResolutionError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputImageResolutionError: [
        r"Resolution not supported by the provider. Use one of the following resolutions: 256x256,512x512"
    ],
}
