from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidPromptError,
    ProviderInvalidInputImageResolutionError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidPromptError: [
        r"The system detected potentially unsafe content\. Please try again or adjust the prompt",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"Resolution not supported by the provider. Use one of the following resolutions: 256x256,512x512"
    ],
}
