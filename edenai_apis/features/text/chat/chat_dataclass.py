from typing import Dict, Generator, Optional, Sequence
from pydantic import BaseModel, StrictStr, Field


class ChatMessageDataClass(BaseModel):
    role: Optional[StrictStr]
    message: Optional[StrictStr]


class ChatDataClass(BaseModel):
    generated_text: StrictStr
    message: Sequence[ChatMessageDataClass] = Field(default_factory=list)

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["generated_text"]

class ChatStreamResponse(BaseModel):
    text: str
    blocked: bool
    provider: str
    
class StreamChat(BaseModel):
    stream: Generator[ChatStreamResponse, None, None]
