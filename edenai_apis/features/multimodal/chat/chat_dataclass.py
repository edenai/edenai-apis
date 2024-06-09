from typing import Dict, Generator, Optional, Sequence, Literal, List
from pydantic import BaseModel, StrictStr, model_validator, Field


class ChatMessageContent(BaseModel):
    media_url: Optional[str] = None
    media_base64: Optional[str] = None
    text: Optional[str] = None
    media_type: Optional[str] = None

    """
    Most of this validation is already done on the backend, it's mostly to check 
    if the message formatting is correct when adding a new provider
    """

    @model_validator(mode="before")
    def _check_content(cls, values):
        media_url, media_base64, text = (
            values.get("media_url"),
            values.get("media_base64"),
            values.get("text"),
        )
        if media_url and (media_base64 or text):
            raise ValueError(
                "If media_url is provided, media_base64 and text should not be provided"
            )
        if media_base64 and (media_url or text):
            raise ValueError(
                "If media_base64 is provided, media_url and text should not be provided"
            )
        if text and (media_url or media_base64):
            raise ValueError(
                "If text is provided, media_url and media_base64 should not be provided"
            )
        return values


class ChatMessage(BaseModel):
    type: Literal["media_url", "media_base64", "text"]
    content: ChatMessageContent

    # @model_validator(mode="before")
    # def check_type_and_content(cls, values):
    #     type_, content = values.get("type"), values.get("content")
    #     if type_ == "media_url" and not content.media_url:
    #         raise ValueError(
    #             "If type is media_url, media_url must be provided in content"
    #         )
    #     if type_ == "media_base64" and not content.media_base64:
    #         raise ValueError(
    #             "If type is media_base64, media_base64 must be provided in content"
    #         )
    #     if type_ == "text" and not content.text:
    #         raise ValueError("If type is text, text must be provided in content")

    #     return values


class ChatMessageDataClass(BaseModel):
    role: Optional[StrictStr]
    content: Sequence[ChatMessage] = Field(default_factory=list)


class ChatDataClass(BaseModel):
    generated_text: StrictStr
    messages: Sequence[ChatMessageDataClass] = Field(default_factory=list)

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["generated_text"]

    @staticmethod
    def generate_standardized_response(
        generated_text: str, messages: List[Dict[str, str]]
    ):
        # Format the messages
        formatted_messages = [
            ChatMessageDataClass(
                role=item["role"],
                content=[
                    ChatMessage(
                        type=sub_item["type"],
                        content=ChatMessageContent(**sub_item["content"]),
                    )
                    for sub_item in item["content"]
                ],
            )
            for item in messages
        ]

        # Append assistant's response
        formatted_messages.append(
            ChatMessageDataClass(
                role="assistant",
                content=[
                    ChatMessage(
                        type="text", content=ChatMessageContent(text=generated_text)
                    )
                ],
            )
        )

        # Construct the standardized response
        standardized_response = ChatDataClass(
            generated_text=generated_text,
            messages=formatted_messages,
        )

        return standardized_response


class ChatStreamResponse(BaseModel):
    text: str
    blocked: bool
    provider: str


class StreamChat(BaseModel):
    stream: Generator[ChatStreamResponse, None, None]
