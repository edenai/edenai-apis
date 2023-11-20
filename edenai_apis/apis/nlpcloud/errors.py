from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInvalidInputPayloadSize,
)

ERRORS: ProviderErrorLists = {
    ProviderInvalidInputPayloadSize: [r"Request Entity Too Large"]
}
