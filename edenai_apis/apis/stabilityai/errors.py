from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderAuthorizationError,
    ProviderInvalidPromptError,
    ProviderInvalidInputImageResolutionError,
    ProviderInternalServerError,
    ProviderLimitationError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderAuthorizationError: [r"Incorrect API key provided"],
    ProviderInvalidPromptError: [r"Invalid prompts detected"],
    ProviderInvalidInputImageResolutionError: [
        r"Resolution not supported by the provider. Use one of the following resolutions: 512x512,1024x1024"
    ],
    ProviderInternalServerError: [
        r"upstream connect error or disconnect/reset before headers. reset reason: connection failure"
    ],
    ProviderLimitationError: [
        r"Your organization does not have enough balance to request this action"
    ],
}
