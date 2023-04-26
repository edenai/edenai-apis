from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputAudioDurationError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputAudioDurationError: [
        r"Audio duration below minimum limit of 2 seconds",
    ]
}
