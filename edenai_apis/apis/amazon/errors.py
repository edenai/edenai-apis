from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderException,
    ProviderInvalidInputAudioDurationError,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputImageResolutionError,
    ProviderInvalidInputPayloadSize,
    ProviderInvalidInputTextLengthError,
    ProviderLimitationError,
    ProviderNotFoundError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderException: [
        r"Client actionable error.",
    ],
    ProviderLimitationError: [
        r"An error occurred \(LimitExceededException\) when calling the CreateVocabulary operation: You have too many vocabularies\. Delete a vocabulary and try your request again",
    ],
    ProviderInvalidInputError: [
        r"An error occurred \(InvalidParameterException\) when calling the \w+ operation: Request has invalid parameters",
        r"Invalid type for parameter LanguageCode",
        r"No face detected in the image",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Invalid length for parameter QueriesConfig.Queries\[\d+\].Text, value: \d+, valid min length: 1",
        r"Max length of request text allowed is \d+ bytes",
    ],
    ProviderInvalidInputFileFormatError: [
        r"An error occurred \(UnsupportedDocumentException\) when calling the \w+ operation: Request has unsupported document format",
        r"An error occurred \(InvalidImageFormatException\) when calling the \w+ operation: Request has invalid image format",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"An error occurred \(ImageTooLargeException\) when calling the \w+ operation: Input image dimensions \d+ x \d+ exceed maximum dimension size of 10000 pixels",
        r"An error occurred \(ImageTooLargeException\) when calling the DetectLabels operation: Image dimensions: null x null pixels exceed the maximum limit.",
    ],
    ProviderInvalidInputFileSizeError: [
        r"An error occurred \(ValidationException\) when calling the \w+ operation: 1 validation error detected: Value '.*' at 'image\.bytes' failed to satisfy constraint: Member must have length less than or equal to \d+",
    ],
    ProviderInvalidInputAudioDurationError: [
        r"The input media file length is too small\. Minimum audio duration is 0\.500000 milliseconds\. Check the length of the file and try your request again\.",
        r"Your audio file must have a speech segment long enough in duration to perform automatic language identification\. Provide an audio file with someone speaking for a longer period of time and try your request again",
    ],
    ProviderNotFoundError: [
        r"An error occurred \(ResourceNotFoundException\) when calling the ListFaces operation: The collection id: .+ does not exist"
    ]
}
