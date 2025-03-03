from edenai_apis.utils.exception import (
    ProviderErrorLists,
    ProviderInternalServerError,
    ProviderTimeoutError,
)

# NOTE: error messages should be regex patterns
ERRORS: ProviderErrorLists = {
    ProviderInternalServerError: [
        r"Error calling Clarifai",
        r"Failure",
    ],
    ProviderTimeoutError: [
        r"<[^<>]+debug_error_string = 'UNKNOWN:Error received from peer ipv4:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+ {created_time:'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[\+\-]\d{2}:\d{2}', grpc_status:14, grpc_message:'GOAWAY received'}'>",
        r"Model is deploying",
    ],
}
