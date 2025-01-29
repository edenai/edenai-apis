import logging
import uuid
from typing import Any, List, Literal, Optional, Type, Union, Dict

from llm_engine.types.response_types import ResponseModel
from llm_engine.clients import LLM_COMPLETION_CLIENTS
from llm_engine.clients.completion import CompletionClient
from llm_engine.utils import format_chat_messages, convert_tools_to_openai
from edenai_apis.utils.types import ResponseType
from edenai_apis.features.text.chat import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    StreamChat,
    ChatStreamResponse,
    ToolCall,
)

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Instantiate the engine for doing calls to LLMs
    """

    def __init__(
        self,
        provider_name: str,
        model: Optional[str] = None,
        client_name: Optional[str] = None,
        application_name: str = uuid.uuid4(),
        provider_config: dict = {},
        **kwargs,
    ) -> None:
        # Set the user
        self.model = model
        self.provider_name = provider_name
        self.application_name = str(application_name)
        if client_name is None:
            client_name = next(iter(LLM_COMPLETION_CLIENTS))
        # TODO change the completion client to behave in the same way
        self.provider_config = provider_config
        self.completion_client: CompletionClient = LLM_COMPLETION_CLIENTS[client_name](
            model_name=model, provider_name=self.provider_name
        )

    def chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
        stream=False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        messages = format_chat_messages(
            text, chatbot_global_action, previous_history, tool_results
        )
        call_params = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "drop_params": True,
        }

        for config_key, config_value in self.provider_config.items():
            call_params[config_key] = config_value

        if available_tools and not tool_results:
            call_params["tools"] = convert_tools_to_openai(available_tools)
            call_params["tool_choice"] = tool_choice

        response = self.completion_client.completion(**call_params, **kwargs)
        response = ResponseModel.model_validate(response)
        if stream is False:
            message = response.choices[0].message
            generated_text = message.content
            original_tool_calls = message.tool_calls or []
            tool_calls = []
            for call in original_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                )
            messages = [
                ChatMessageDataClass(role="user", message=text, tools=available_tools),
                ChatMessageDataClass(
                    role="assistant",
                    message=generated_text,
                    tool_calls=tool_calls,
                ),
            ]
            messages_json = [m.dict() for m in messages]

            standardized_response = ChatDataClass(
                generated_text=generated_text, message=messages_json
            )

            return ResponseType[ChatDataClass](
                original_response=response,
                standardized_response=standardized_response,
                usage=response.usage,
            )
        else:
            stream = (
                ChatStreamResponse(
                    text=chunk.to_dict()["choices"][0]["delta"].get("content", ""),
                    blocked=not chunk.to_dict()["choices"][0].get("finish_reason")
                    in (None, "stop"),
                    provider="openai",
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamChat](
                original_response=None, standardized_response=StreamChat(stream=stream)
            )
