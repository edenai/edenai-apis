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
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Timeout while synthesizing",
    ],
    ProviderParsingError: [
        r"No table found in the document.",
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
    ],
    ProviderInvalidInputFileError: [
        r"Input data is not a valid image",
        r"The target language is not valid",
        r"The file is corrupted or format is unsupported",
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
}
