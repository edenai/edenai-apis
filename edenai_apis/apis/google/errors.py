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
        r"\d*\S*Invalid number of target languages per request",
        r"\d*\S*Request contains an invalid argument.",
        r"\d*\S*Requested language code \w+ doesn't match the voice \w+ language code \w+. Either pick a different voice, or change the requested language code to \w+.",
        r"\d*\S*Target language can't be equal to source language. \w+",
        r"The language \w+ is not supported for \w+.",
        r"\d*\S*Invalid SSML. Newer voices like \w+ require valid SSML"
        r"Wrong voice id",
    ],
    ProviderInvalidInputTextLengthError: [
        r"\d*\S*Either `input\.text` or `input\.ssml` is longer than the limit of 5000 bytes",
        r"\d*\S*This request contains sentences that are too long",
        r"\d*\S*Text is too long",
        r"Invalid text content: too few tokens \(words\) to process",
        r"\d*\S*Input files contain a sentence larger than max_size, \d+ > \d+",
        r"\d*\S*This request contains sentences that are too long. Consider splitting up long sentences with sentence ending punctuation e.g. periods. Also consider removing SSML sentence tags (e.g. '<s>') as they can confuse Cloud Text-to-Speech.",
        r"\d*\S*This request contains sentences that are too long. To fix, split up long sentences with sentence ending punctuation e.g. periods.",
    ],
    ProviderInvalidInputFileError: [
        r"Bad image data",
        r"Invalid argument: Bad image data",
        r"\d*\S*Invalid image content",
        r"\d*\S*No result error, PDF may be invalid.",
        r"cannot identify image file \w+",
    ],
    ProviderInvalidInputPayloadSize: [
        r"\d*\S*Request payload size exceeds the limit:",
    ],
    ProviderInvalidInputFileSizeError: [
        r"Document size \(\d+\) exceeds the limit: 20971520",
        r"\d*\S*Input size limit exceeded for Studio Voice.",
        r"\d*\S*The document is larger than the maximum size of 1000000 bytes.",
    ],
    ProviderInvalidInputDocumentPages: [
        r"\d*\S*Exceed the maximum PDF page support. Received: \d+. Support up to: 20",
        r"\d*\S*Document pages exceed the limit: 15 got \d+ \w+",
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
        r"\d*\S*sample_rate_hertz \(\d+\) in \w+ must either be omitted or match the value in the WAV header \(\d+\).",
    ],
    ProviderInvalidInputFileFormatError: [
        r"File extension not supported. Use one of the following extensions: \w+",
        r"Provider google doesn't support file type: \w+ for this feature. Supported mimetypes are \w+",
        r"\d*\S*Unsupported input file format.",
        r"\d*\S*WAV header indicates an unsupported format.",
        r"Audio format not supported. Use one of the following: \w+",
    ],
}
