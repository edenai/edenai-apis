from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputAudioEncodingError,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputPayloadSize,
    ProviderMissingInputError,
    ProviderNotFoundError,
    ProviderParsingError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderMissingInputError: [
        r"The parameter 'target' should not be empty",
    ],
    ProviderParsingError: [
        r"Unable to detect the source language",
        r"Mismatched tag found by parser",
    ],
    ProviderInvalidInputPayloadSize: [
        r"Payload length \d+ exceeds the payload limit 5120 bytes.",
        r"Exceeded max content length for translate request, max allowed is 50 KiB",
        r"Error: Payload length \d+ exceeds the payload limit 5120 bytes., Code: 400 , Information: {'code_description': 'Bad Request'} , X-dp-watson-tran-id: [a-f0-9\-]+ , X-global-transaction-id: [a-f0-9\-]+",
    ],
    ProviderInvalidInputError: [
        r"not enough text for language id",
        r"unsupported text language: \w+",
        r"Input contains unmatched open SSML tags",
        r"Sampling rate must lie in the range of 8 kHz to 192 kHz",
        r"\w+ with attribute volume is not supported in the current voice",
        r"Only \w+ voice is available for the \w+ language code",
    ],
    ProviderInternalServerError: [
        r"Internal Server Error",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Audio format not supported. Use one of the following: \w+",
        r"File extension not supported. Use one of the following extensions: \w+",
    ],
    ProviderInvalidInputAudioEncodingError: [
        r"unable to transcode data stream \w+ -> \w+",
    ],
    ProviderNotFoundError: [
        r"Error: Unable to find model for specified languages, Code: 404 , X-dp-watson-tran-id: [a-f0-9\-]+ , X-global-transaction-id: [a-f0-9\-]+",
        r"Error: Model \w+ not found, Code: 404 , Information: {'code_description': 'Not Found'} , X-dp-watson-tran-id: [a-f0-9\-]+ , X-global-transaction-id: [a-f0-9\-]+",
    ],
}
