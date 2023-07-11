from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderInvalidInputPayloadSize,
    ProviderInvalidInputTextLengthError,
    ProviderLimitationError,
    ProviderParsingError,
    ProviderInvalidInputAudioDurationError,
    ProviderInvalidInputFileFormatError,
    ProviderAuthorizationError,
    ProviderInvalidPromptError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"The server had an error while processing your request\. Sorry about that",
        r"Bad gateway.",
        r"Gateway timeout",
        r"Internal error",
        r"Internal Server Error",
        r"Request failed due to server shutdown",
    ],
    ProviderParsingError: [
        r"An error occurred while parsing the response",
    ],
    ProviderLimitationError: [
        r"You exceeded your current quota, please check your plan and billing details.",
        r"Billing hard limit has been reached",
        r"Rate limit exceeded for images per minute",
        r"Rate limit reached for \w+ in organization org-TX8lX5CiTdnObb5wxAAmeZZU on tokens per min. Limit: \d+ / min. Please try again in 1ms. Contact us through our help center at help.openai.com if you continue to have issues.",
        r"Rate limit reached for default-whisper-1 in organization org-TX8lX5CiTdnObb5wxAAmeZZU on requests per min. Limit: 50 / min."
        r"Too many parallel completions requested. You submitted \d+ prompts, but you can currently request up to at most a total of \d+\). Please contact us through our help center at help.openai.com for further questions. \(HINT: if you want to just evaluate probabilities without generating new text, you can submit more prompts if you set 'max_tokens' to 0.\)",
        r"That model is currently overloaded with other requests. You can retry your request, or contact us through our help center at help.openai.com if the error persists",
    ],
    ProviderInvalidInputTextLengthError: [
        r"This model's maximum context length is \d+ tokens",
    ],
    ProviderInvalidInputPayloadSize: [
        r"Maximum content size limit \(26214400\) exceeded \(\d+ bytes read\)",
    ],
    ProviderInvalidInputError: [
        r"Your request was rejected as a result of our safety system",
        r"Invalid language format for: \w+.",
        r"Wrong model name, availaible models for openai are : \w+",
    ],
    ProviderInvalidInputAudioDurationError: [
        r"Audio file is too short. Minimum audio length is 0.1 seconds."
    ],
    ProviderInvalidInputFileFormatError: [
        r"File extension not supported. Use one of the following extensions: \w+"
    ],
    ProviderAuthorizationError: [
        r"You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY), or as the password field (with blank username) if you're accessing the API from your browser and are prompted for a username and password. You can obtain an API key from https://platform.openai.com/account/api-keys"
    ],
    ProviderInvalidPromptError: [
        r"Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our safety system."
    ],
}
