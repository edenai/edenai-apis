from ..chat.chat_dataclass import *
from typing import AsyncGenerator
from pydantic import BaseModel, ConfigDict


class StreamAchat(BaseModel):
    stream: AsyncGenerator[ModelResponseStream, None]

    model_config = ConfigDict(arbitrary_types_allowed=True)
