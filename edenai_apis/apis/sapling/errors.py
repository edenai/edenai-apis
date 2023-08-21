# NOTE: error messages should be regex patterns
from edenai_apis.utils.exception import (
    ProviderErrorLists, 
    ProviderInternalServerError
)


ERRORS: ProviderErrorLists = {
     ProviderInternalServerError: [
         r"Internal server error"
     ]
}