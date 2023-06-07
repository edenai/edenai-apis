from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderInvalidInputPayloadSize,
    ProviderMissingInputError,
    ProviderParsingError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderMissingInputError: [
        r"The parameter 'target' should not be empty",
    ],
    ProviderParsingError: [
        r"Unable to detect the source language",
    ],
    ProviderInvalidInputPayloadSize: [
        r"Payload length \d+ exceeds the payload limit 5120 bytes.",
        r"Exceeded max content length for translate request, max allowed is 50 KiB",
    ],
    ProviderInvalidInputError: [
        r"unsupported text language: unknown",
        r"Input contains unmatched open SSML tags",
    ],
    ProviderInternalServerError: [
        r"Internal Server Error",
    ],
}
