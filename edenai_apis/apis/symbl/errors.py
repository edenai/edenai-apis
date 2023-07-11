from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputError,
    ProviderInvalidInputFileFormatError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInvalidInputError: [
        r"Job with .+ not found",
    ],
    ProviderInvalidInputFileFormatError: [
        r"Detected MIME Type as \\'video/mp4\\'. This endpoint only supports audio files. For video files please use /v1/process/video or /v1/process/video/url endpoints.",
        r"File extension not supported. Use one of the following extensions: \w+",
    ],
}
