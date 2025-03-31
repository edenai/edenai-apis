from ibm_watson import ApiException

from edenai_apis.utils.exception import (
    AsyncJobException,
    AsyncJobExceptionReason,
    ProviderException,
)


def handle_ibm_call(function_call, **kwargs):
    provider_job_id_error = "job not found"
    try:
        response = function_call(**kwargs)
    except ApiException as exc:
        message = exc.message
        code = exc.code
        if provider_job_id_error in str(exc):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID, code=code
            )
        raise ProviderException(message, code=code)
    except Exception as exc:
        if provider_job_id_error in str(exc):
            raise AsyncJobException(reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID)
        raise ProviderException(str(exc))
    return response
