from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputAudioEncodingError,
    ProviderInvalidInputDocumentPages,
    ProviderInvalidInputError,
    ProviderInvalidInputFileError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputPayloadSize,
    ProviderInvalidInputTextLengthError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Request contains an invalid argument",
        r"400 Invalid number of target languages per request",
        r"400 Request contains an invalid argument.",
        r"400 Requested language code \w+ doesn't match the voice \w+ language code \w+. Either pick a different voice, or change the requested language code to \w+.",
        r"400 Target language can't be equal to source language. \w+",
        r"The language \w+ is not supported for \w+.",
        r"400 Invalid SSML. Newer voices like \w+ require valid SSML" r"Wrong voice id",
    ],
    ProviderInvalidInputTextLengthError: [
        r"400 Either `input\.text` or `input\.ssml` is longer than the limit of 5000 bytes",
        r"400 This request contains sentences that are too long",
        r"400 Text is too long",
        r"Invalid text content: too few tokens \(words\) to process",
        r"400 Input files contain a sentence larger than max_size, \d+ > \d+",
        r"400 This request contains sentences that are too long. Consider splitting up long sentences with sentence ending punctuation e.g. periods. Also consider removing SSML sentence tags (e.g. '<s>') as they can confuse Cloud Text-to-Speech.",
        r"400 This request contains sentences that are too long. To fix, split up long sentences with sentence ending punctuation e.g. periods.",
    ],
    ProviderInvalidInputFileError: [
        r"Bad image data",
        r"Invalid argument: Bad image data",
        r"400 Invalid image content",
        r"400 No result error, PDF may be invalid.",
        r"cannot identify image file \w+",
    ],
    ProviderInvalidInputPayloadSize: [
        r"400 Request payload size exceeds the limit:",
    ],
    ProviderInvalidInputFileSizeError: [
        r"Document size \(\d+\) exceeds the limit: 20971520",
        r"400 Input size limit exceeded for Studio Voice.",
        r"400 The document is larger than the maximum size of 1000000 bytes.",
    ],
    ProviderInvalidInputDocumentPages: [
        r"400 Exceed the maximum PDF page support. Received: \d+. Support up to: 20",
        r"400 Document pages exceed the limit: 15 got \d+ \w+",
    ],
    ProviderInternalServerError: [
        r"500 Internal error encountered",
        r"500 Attempts is nullptr",
        r"500 Exception deserializing response!",
        r"Internal server error. Unexpected feature response.",
        r"The service is currently unavailable",
        r"500 Received RST_STREAM with error code 2",
        r"503 502:Bad Gateway",
        r"Deadline of \d+.0s exceeded while calling target function, last exception: 503 The service is currently unavailable.",
        r"Internal server error. Failed to process features.",
    ],
    ProviderInvalidInputAudioEncodingError: [
        r"bad encoding",
        r"Could not decode audio file, bad file encoding",
        r"400 sample_rate_hertz \(\d+\) in \w+ must either be omitted or match the value in the WAV header \(\d+\).",
    ],
    ProviderInvalidInputFileFormatError: [
        r"File extension not supported. Use one of the following extensions: \w+",
        r"Provider google doesn't support file type: \w+ for this feature. Supported mimetypes are \w+",
        r"400 Unsupported input file format.",
        r"400 WAV header indicates an unsupported format.",
        r"Audio format not supported. Use one of the following: \w+",
    ],
}
