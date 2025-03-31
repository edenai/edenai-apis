from typing import List, Optional, Union, Dict, Any, Literal, Generator
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from litellm import ModelResponseStream


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: ChatRole = Field(..., description="The role of the message author")
    content: Optional[str] = Field(None, description="The content of the message")
    name: Optional[str] = Field(
        None, description="The name of the author of this message"
    )

    # For function calls
    function_call: Optional[Dict[str, Any]] = Field(
        None, description="The function call information"
    )

    # For tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="The tool call information"
    )


class ChatCompletionModel(str, Enum):
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_VISION = "gpt-4-vision-preview"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"


class FunctionDefinition(BaseModel):
    name: str = Field(..., description="The name of the function to be called")
    description: Optional[str] = Field(
        None, description="A description of what the function does"
    )
    parameters: Dict[str, Any] = Field(
        ..., description="The parameters the function accepts, in JSON Schema format"
    )


class ToolDefinition(BaseModel):
    type: Literal["function"] = Field("function", description="The type of tool")
    function: FunctionDefinition = Field(..., description="The function definition")


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = Field(
        "text", description="The format of the response"
    )


class ChatCompletionRequest(BaseModel):
    model: Union[ChatCompletionModel, str] = Field(
        ..., description="ID of the model to use"
    )
    messages: List[ChatMessage] = Field(
        ..., description="A list of messages comprising the conversation so far"
    )
    functions: Optional[List[FunctionDefinition]] = Field(
        None, description="A list of functions the model may generate JSON inputs for"
    )
    tools: Optional[List[ToolDefinition]] = Field(
        None, description="A list of tools the model may use"
    )
    function_call: Optional[Union[str, Dict[str, str]]] = Field(
        None, description="Controls how the model responds to function calls"
    )
    temperature: Optional[float] = Field(
        1.0, description="What sampling temperature to use, between 0 and 2", ge=0, le=2
    )
    top_p: Optional[float] = Field(
        1.0,
        description="An alternative to sampling with temperature, called nucleus sampling",
        ge=0,
        le=1,
    )
    n: Optional[int] = Field(
        1,
        description="How many chat completion choices to generate for each input message",
    )
    stream: Optional[bool] = Field(
        False, description="If set, partial message deltas will be sent"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens",
    )
    max_tokens: Optional[int] = Field(
        None,
        description="The maximum number of tokens to generate in the chat completion",
    )
    presence_penalty: Optional[float] = Field(
        0,
        description="Number between -2.0 and 2.0 to penalize tokens based on their presence so far",
        ge=-2.0,
        le=2.0,
    )
    frequency_penalty: Optional[float] = Field(
        0,
        description="Number between -2.0 and 2.0 to penalize tokens based on their frequency so far",
        ge=-2.0,
        le=2.0,
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None,
        description="Modify the likelihood of specified tokens appearing in the completion",
    )
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )
    response_format: Optional[ResponseFormat] = Field(
        None, description="An object specifying the format that the model must output"
    )
    seed: Optional[int] = Field(None, description="A seed for deterministic sampling")

    @model_validator(mode="after")
    def check_functions_and_tools(cls, values):
        functions = values.get("functions")
        tools = values.get("tools")

        if functions is not None and tools is not None:
            raise ValueError("You cannot provide both 'functions' and 'tools'")

        return values


class ToolCallFunction(BaseModel):
    name: str = Field(..., description="The name of the function to call")
    arguments: str = Field(
        ..., description="The arguments to call the function with, as a JSON string"
    )


class ToolCall(BaseModel):
    id: str = Field(..., description="The ID of the tool call")
    type: Literal["function"] = Field(..., description="The type of tool call")
    function: ToolCallFunction = Field(..., description="The function to call")


class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="The index of this completion choice")
    message: ChatMessage = Field(..., description="The chat completion message")
    finish_reason: str = Field(
        ...,
        description="The reason the completion finished: 'stop', 'length', 'tool_calls', 'content_filter', or 'function_call'",
    )


class UsageTokensDetails(BaseModel):
    audio_tokens: Optional[int] = Field(
        ..., description="Number of audio tokens in the prompt"
    )
    cached_tokens: Optional[int] = Field(
        ..., description="Number of cached tokens in the prompt"
    )
    prompt_tokens: Optional[int] = Field(
        ..., description="Number of tokens in the prompt"
    )
    completion_tokens: Optional[int] = Field(
        ..., description="Number of tokens in the generated completion"
    )
    total_tokens: Optional[int] = Field(
        ..., description="Total number of tokens used (prompt + completion)"
    )
    accepted_prediction_tokens: Optional[int] = Field(
        ..., description="Number of accepted tokens in the prompt"
    )
    reasoning_tokens: Optional[int] = Field(
        ..., description="Number of reasoning tokens in the prompt"
    )
    rejected_prediction_tokens: Optional[int] = Field(
        ..., description="Number of rejected tokens in the prompt"
    )


class ChatCompletionUsage(BaseModel):
    completion_tokens_details: Optional[UsageTokensDetails] = Field(
        ..., description="Number of tokens in the generated completion"
    )
    prompt_tokens_details: Optional[UsageTokensDetails] = Field(
        ..., description="Number of tokens in the prompt"
    )
    total_tokens: int = Field(
        ..., description="Total number of tokens used (prompt + completion)"
    )


class ChatDataClass(BaseModel):
    id: str = Field(..., description="Unique identifier for this completion")
    object: str = Field(..., description="Object type, always 'chat.completion'")
    created: int = Field(
        ..., description="Unix timestamp for when the completion was created"
    )
    model: str = Field(..., description="The model used for completion")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="List of chat completion choices generated by the model"
    )
    usage: ChatCompletionUsage = Field(
        ..., description="Usage statistics for the completion request"
    )
    system_fingerprint: Optional[str] = Field(
        None, description="Identifier for the system version that processed the request"
    )


class StreamChat(BaseModel):
    stream: Generator[ModelResponseStream, None, None]
