from typing import Dict, Generator, List, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class ToolCall(BaseModel):
    id: StrictStr
    name: StrictStr
    arguments: StrictStr


class ChatMessageDataClass(BaseModel):
    role: Optional[StrictStr]
    message: Optional[StrictStr] = ""
    tools: Optional[List[dict]] = Field(
        default=None, description="Tools defined by the user"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="The tools arguments generated from tools definition and user prompt.",
    )


class ChatDataClass(BaseModel):
    generated_text: Optional[StrictStr]
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
