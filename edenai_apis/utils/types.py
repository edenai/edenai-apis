from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import StrictStr, BaseModel
from pydantic.generics import GenericModel

T = TypeVar('T')

ResponseSuccess = TypeVar("ResponseSuccess")
SUCCESS = {"status": "success"}

class ResponseType(GenericModel, Generic[T]):
    original_response: Any
    standarized_response: T

class AsyncLaunchJobResponseType(BaseModel):
    provider_job_id: StrictStr

class AsyncBaseResponseType(GenericModel, Generic[T]):
    status: StrictStr
    provider_job_id: StrictStr

class AsyncPendingResponseType(AsyncBaseResponseType, GenericModel, Generic[T]):
    status: StrictStr = "pending"

class AsyncErrorResponseType(AsyncBaseResponseType, GenericModel, Generic[T]):
    status: StrictStr = "failed"
    error: Optional[Dict]

class AsyncResponseType(ResponseType, AsyncBaseResponseType, GenericModel, Generic[T]):
    status: StrictStr = "succeeded"
