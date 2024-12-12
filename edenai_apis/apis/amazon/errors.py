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
        r"Invalid language format for: \w+.",
        r"Only \w+ voice is available for the \w+ language code",
        r"Wrong voice id",
        r"An error occurred \(InvalidParameterException\) when calling the \w+ operation: Request has invalid parameters",
        r"Invalid type for parameter LanguageCode",
        r"No face detected in the image",
        r"Provider does not support selected (?:target|source_)?language: \w+",
        r"This provider doesn't auto-detect languages, please provide a valid \w+",
        r"An error occurred \(InvalidParameterException\) when calling the \w+ operation: There are no faces in the image. Should be at least 1.",
        r"An error occurred \(ValidationException\) when calling the \w+ operation: Value \w+ at 'languageCode'failed to satisfy constraint: Member must satisfy enum value set: [de, pt, en, it, fr, es]",
        r"An error occurred \(BadRequestException\) when calling the StartTranscriptionJob operation: 1 validation error detected: Value '\d+' at 'settings.maxSpeakerLabels' failed to satisfy constraint: Member must have value less than or equal to \d+",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Invalid length for parameter QueriesConfig.Queries\[\d+\].Text, value: \d+, valid min length: 1",
        r"Max length of request text allowed is \d+ bytes",
        r"An error occurred \(TextLengthExceededException\) when calling the \w+ operation: Maximum text length has been exceeded",
        r"An error occurred \(TextSizeLimitExceededException\) when calling the \w+ operation: Input text size exceeds limit. Max length of request text allowed is 100000 bytes while in this request the text size is \d+ bytes",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Request has unsupported document format",
        r"Request has invalid image format",
        r"Audio format not supported. Use one of the following: \w+",
        r"Provider amazon doesn't support file type: \w+ for this feature. Supported mimetypes are \w+",
        r"File extension not supported. Use one of the following extensions: \w+",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"An error occurred \(ImageTooLargeException\) when calling the \w+ operation: Input image dimensions \d+ x \d+ exceed maximum dimension size of 10000 pixels",
        r"An error occurred \(ImageTooLargeException\) when calling the DetectLabels operation: Image dimensions: null x null pixels exceed the maximum limit.",
    ],
    ProviderInvalidInputFileSizeError: [
        r"An error occurred \(ValidationException\) when calling the \w+ operation: 1 validation error detected: Value '.*' at 'image\.bytes' failed to satisfy constraint: Member must have length less than or equal to \d+",
    ],
    ProviderInvalidInputAudioDurationError: [
        r"The input media file length is too small. Minimum audio duration is 0.500000 seconds. Check the length of the file and try your request again",
        r"The input media file length is too small\. Minimum audio duration is 0\.500000 milliseconds\. Check the length of the file and try your request again\.",
        r"Your audio file must have a speech segment long enough in duration to perform automatic language identification\. Provide an audio file with someone speaking for a longer period of time and try your request again",
    ],
    ProviderNotFoundError: [
        r"An error occurred \(ResourceNotFoundException\) when calling the ListFaces operation: The collection id: .+ does not exist",
        r"Face Collection is empty",
    ],
    ProviderInvalidInputPayloadSize: [
        r"An error occurred (413) when calling the \w+ operation:",
    ],
}
