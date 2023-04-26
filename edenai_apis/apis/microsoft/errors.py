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
        r"Face List number reached limit\. \(Parameter 'faceListCount'\)",
    ],
    ProviderInvalidInputError: [
        r"Invalid Document in request",  # Document not referring to a file (appears in text features)
        r"Job task: 'ExtractiveSummarization' failed with validation error: Job task parameter value '\d+' is not supported for sentenceCount parameter for job task type ExtractiveSummarization. Supported values 1 \(min\) to 20 \(max\).",
        r"There is more than 1 face in the image.",
    ],
    ProviderInvalidInputFileError: [
        r"Input data is not a valid image",
    ],
    ProviderInvalidInputFileSizeError: [
        r"Input image is too large",
        r"Image size is too small.",
        r"Image size is too big. The valid image file size should be no larger than 6MB.",
    ],
    ProviderInvalidInputImageResolutionError: [
        r"The height or width of the image is outside the supported range",
        r"Image must be at least 50 pixels in width and height",
    ],
    ProviderInvalidInputTextLengthError: [
        r"Text is too long for spell check\. Max length is 130 characters",
    ],
}
