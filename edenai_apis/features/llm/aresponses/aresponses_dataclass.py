from ..responses.responses_dataclass import *
from typing import AsyncGenerator
from pydantic import BaseModel, ConfigDict


class StreamAResponses(BaseModel):
    stream: AsyncGenerator

    model_config = ConfigDict(arbitrary_types_allowed=True)
