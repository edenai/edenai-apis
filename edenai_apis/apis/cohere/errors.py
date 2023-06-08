from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderInvalidInputError,
    ProviderInvalidInputTextLengthError,
    ProviderLimitationError,
    ProviderParsingError,
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
        r"Error calling Clarifai API",
        r"Failure",
    ],
    ProviderInvalidInputError: [
        r"blocked output: please adjust your prompt and try again, as this generation may be a potential violation of our Usage Guidelines \(https://docs.cohere.ai/usage-guidelines/\).",
    ],
    ProviderInvalidInputTextLengthError: [
        r"text must be longer than 250 characters",
    ],
}
