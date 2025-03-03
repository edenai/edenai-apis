from edenai_apis.utils.exception import ProviderErrorLists, ProviderInternalServerError

ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [r".*Internal server error.*"],
}
