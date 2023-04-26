from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderException,
    ProviderInvalidInputAudioDurationError,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputImageResolutionError,
    ProviderInvalidInputTextLengthError,
    ProviderLimitationError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderException: [
        r"Client actionable error.",
    ],
    ProviderLimitationError: [
        r"An error occurred (LimitExceededException) when calling the CreateVocabulary operation: You have too many vocabularies\. Delete a vocabulary and try your request again",
    ],
    ProviderInvalidInputError: [
        r"An error occurred (InvalidParameterException) when calling the \w operation: Request has invalid parameters",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Invalid length for parameter QueriesConfig.Queries\[\d+\].Text, value: \d+, valid min length: 1",
        r"An error occurred (TextSizeLimitExceededException) when calling the \w operation: Input text size exceeds limit. Max length of request text allowed is 100000 bytes while in this request the text size is \d+ bytes",
        r"An error occurred (TextLengthExceededException) when calling the \w operation: Maximum text length has been exceeded",
    ],
    ProviderInvalidInputFileFormatError: [
        r"An error occurred (UnsupportedDocumentException) when calling the \w operation: Request has unsupported document format",
        r"An error occurred (InvalidImageFormatException) when calling the \w operation: Request has invalid image format",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"An error occurred (ImageTooLargeException) when calling the \w operation: Input image dimensions \d+ x \d+ exceed maximum dimension size of 10000 pixels",
        r"An error occurred (ImageTooLargeException) when calling the DetectLabels operation: Image dimensions: null x null pixels exceed the maximum limit.",
    ],
    ProviderInvalidInputFileSizeError: [
        r"An error occurred (ValidationException) when calling the \w operation: 1 validation error detected: Value 'java\.nio\.HeapByteBuffer\[pos=0 lim=5357237 cap=5357237\]' at 'image\.bytes' failed to satisfy constraint: Member must have length less than or equal to 5242880",
    ],
    ProviderInvalidInputAudioDurationError: [
        r"The input media file length is too small\. Minimum audio duration is 0\.500000 milliseconds\. Check the length of the file and try your request again\.",
    ],
}
