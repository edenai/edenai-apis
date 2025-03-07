from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import StrictStr, BaseModel

T = TypeVar("T")


class ResponseSuccess(BaseModel):
    status: StrictStr = "success"


class ResponseType(BaseModel, Generic[T]):
    original_response: Any
    standardized_response: T
    usage: Optional[Any] = None
    cost: Optional[Any] = None


class AsyncLaunchJobResponseType(BaseModel):
    provider_job_id: StrictStr


class AsyncBaseResponseType(BaseModel, Generic[T]):
    status: StrictStr
    provider_job_id: StrictStr


class AsyncPendingResponseType(AsyncBaseResponseType, Generic[T]):
    status: StrictStr = "pending"


class AsyncErrorResponseType(AsyncBaseResponseType, Generic[T]):
    status: StrictStr = "failed"
    error: Optional[Dict]


class AsyncResponseType(ResponseType, AsyncBaseResponseType, Generic[T]):
    status: StrictStr = "succeeded"
