from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputAudioEncodingError,
    ProviderInvalidInputDocumentPages,
    ProviderInvalidInputFileError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputPayloadSize,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputTextLengthError: [
        r"400 Either `input\.text` or `input\.ssml` is longer than the limit of 5000 bytes",
        r"400 This request contains sentences that are too long",
        r"400 Text is too long"
    ],
    ProviderInvalidInputFileError: [
        r"Bad image data",
    ],
    ProviderInvalidInputPayloadSize: [
        r"400 Request payload size exceeds the limit:",
    ],
    ProviderInvalidInputFileSizeError: [
        r"Document size (\d+) exceeds the limit: 20971520"
    ],
    ProviderInvalidInputDocumentPages: [
        r"400 Exceed the maximum PDF page support. Received: \d+. Support up to: 20",
        r"400 Document pages exceed the limit: 15 got \d+ "
    ],
    ProviderInternalServerError: [
        r"500 Internal error encountered"
        r"Internal server error\. Unexpected feature response"
    ],
    ProviderInvalidInputAudioEncodingError: [
        r"bad encoding"
        r"Could not decode audio file, bad file encoding",
    ],
}
