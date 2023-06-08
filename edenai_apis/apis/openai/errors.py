from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderInvalidInputPayloadSize,
    ProviderInvalidInputTextLengthError,
    ProviderLimitationError,
    ProviderParsingError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"The server had an error while processing your request\. Sorry about that",
    ],
    ProviderParsingError: [
        r"An error occurred while parsing the response",
    ],
    ProviderLimitationError: [
        r"You exceeded your current quota, please check your plan and billing details.",
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
    ]
}
