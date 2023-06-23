from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputAudioDurationError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputError
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputAudioDurationError: [
        r"Audio duration below minimum limit of 2 seconds",
    ],
    ProviderInvalidInputError : [
        r"filter_profanity: This option is not allowed for foreign languages. Use \w+ language for profanity filter"
    ],
    ProviderInvalidInputFileFormatError : [
        r"File extension not supported. Use one of the following extensions: \w+"
    ]
}
