from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderInvalidInputFileError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputImageResolutionError,
    ProviderInvalidInputTextLengthError,
    ProviderLimitationError,
    ProviderParsingError,
    ProviderTimeoutError,
    ProviderNotFoundError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputPayloadSize,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Timeout while synthesizing",
    ],
    ProviderParsingError: [
        r"No table found in the document.",
        r"Connection was closed by the remote host. Error code: 1007. Error details: \w+ is an unexpected token. The expected token is \w+ or \w+. Line \d+, position \d+. USP state: 3. Received audio size: \w+",
    ],
    ProviderLimitationError: [
        r"Face List number reached limit",
        r"Error calling Microsoft Api: Requests to the Face - Detect Operation under Face API - v1.0 have exceeded call rate limit of your current Face S0 pricing tier. Please retry after \d+ second. Please contact Azure support service if you would like to further increase the default rate limit",
    ],
    ProviderInvalidInputError: [
        r"Invalid Document in request",  # 'Document' does not refer to a file (error appears in text features)
        r"Job task: 'ExtractiveSummarization' failed with validation error: Job task parameter value '\d+' is not supported for sentenceCount parameter for job task type ExtractiveSummarization. Supported values 1 \(min\) to 20 \(max\).",
        r"There is more than 1 face in the image.",
        r"Ssml should only contain one language",
        r"Expected 1-8 alphanumeric characters, got \w+"
        r"The target language is not valid",
        r"No face detected in the image",
        r"Remove audio attributes \w+ to be able to use ssml tags, or add them manually using tags.",
        r"The input language is not supported.",
        r"Wrong voice id",
    ],
    ProviderInvalidInputFileError: [
        r"Input data is not a valid image",
        r"The file is corrupted or format is unsupported",
        r"cannot identify image file \w+",
    ],
    ProviderInvalidInputFileSizeError: [
        r"Input image is too large",
        r"Image size is too small.",
        r"Image size is too big. The valid image file size should be no larger than 6MB.",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"The height or width of the image is outside the supported range",
        r"Image must be at least 50 pixels in width and height",
        r"The input image dimensions are out of range",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Text is too long for spell check\. Max length is 130 characters",
    ],
    ProviderTimeoutError: [
        r"The operation was timeout.",
        r"USP error: timeout waiting for the first audio chunk",
    ],
    ProviderNotFoundError: [
        r"Face list is not found. \(Parameter 'faceListId'\)",
        r"Persisted face [a-z0-9\-]+ is not found. \(Parameter 'persistedFaceId'\)",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Decoding error, image format unsupported.",
        r"File extension not supported. Use one of the following extensions: \w+",
    ],
    ProviderInvalidInputPayloadSize: [r"The maximum request size has been exceeded."],
}
