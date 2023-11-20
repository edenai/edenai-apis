from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidPromptError,
    ProviderLimitationError,
    ProviderParsingError,
    ProviderInvalidInputError,
    ProviderInvalidInputTextLengthError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderLimitationError: [
        r"You are using a Trial key, which is limited to 50 API calls / month. You can continue to use the Trial key for free or upgrade to a Production key with higher rate limits at 'https://dashboard.cohere.ai/api-keys'. Contact us on 'https://discord.gg/XW44jPfYJu' or email us at support@cohere.com with any questions",
    ],
    ProviderParsingError: [
        r"Provider has not found a sentiment of the text.",
        r"An error occurred while parsing the response",
    ],
    ProviderInternalServerError: [
        r"Error calling Cohere API",
        r"Internal Server Error",
        r"Failure",
        r"internal server error, this has been reported to our developer",
    ],
    ProviderInvalidPromptError: [
        r"blocked output: please adjust your prompt and try again, as this generation may be a potential violation of our Usage Guidelines \(https://docs.cohere.ai/usage-guidelines/\).",
        r"blocked input: please adjust your prompt and try again, as it may be a potential violation of our Usage Guidelines \(https://docs.cohere.ai/usage-guidelines/\).",
    ],
    ProviderInvalidInputTextLengthError: [
        r"text must be longer than 250 characters",
    ],
    ProviderInvalidInputError: [
        r"Invalid language format for: \w+",
        r"invalid request: each unique label must have at least 2 examples. Not enough examples for: \w+",
        r"Wrong model name, availaible models for cohere are : command, command-light, command-nightly, command-light-nightly, base, base-light",
    ],
    ProviderInvalidInputTextLengthError: [
        r"invalid request: text must be longer than 250 characters",
        r"invalid request: text size limit exceeded by \d+ characters.",
        r"too many tokens: total number of tokens \(prompt and prediction\) cannot exceed 2048 - received \d+. Try using a shorter prompt, a smaller max_tokens value, or enabling prompt truncating. See https://docs.cohere.ai/reference/generate for more details",
    ],
}
