import json
import httpx
import base64
from typing import List, Optional, Dict


class Mappings:
    @staticmethod
    def format_chat_messages(
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        tool_results: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Formats messages for chat completion API."""
        messages = []
        previous_history = previous_history or []

        for msg in previous_history:
            message = {
                "role": msg.get("role"),
                "content": msg.get("message"),
            }
            if msg.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": tool["id"],
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "arguments": tool["arguments"],
                        },
                    }
                    for tool in msg["tool_calls"]
                ]
            messages.append(message)

        if text and not tool_results:
            messages.append({"role": "user", "content": text})

        if tool_results:
            for tool in tool_results or []:
                tool_call = Mappings.get_tool_call_from_history_by_id(
                    tool["id"], previous_history
                )
                try:
                    result = json.dumps(tool["result"])
                except json.JSONDecodeError:
                    result = str(tool["result"])
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"],
                    }
                )
        if chatbot_global_action:
            messages.insert(0, {"role": "system", "content": chatbot_global_action})

        return messages

    @staticmethod
    def get_tool_call_from_history_by_id(
        tool_call_id: str, previous_history: List[Dict[str, str]]
    ):
        """Helper function to find a tool call in history by its ID."""
        for msg in previous_history:
            if msg.get("tool_calls"):
                for tool in msg["tool_calls"]:
                    if tool["id"] == tool_call_id:
                        return tool
        return None

    @staticmethod
    def convert_tools_to_openai(tools: Optional[List[dict]]):
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            openai_tools.append({"type": "function", "function": tool})
        return openai_tools

    @staticmethod
    def format_multimodal_messages(
        messages: List[any], system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        This funciton comes directly from the edenai_apis package. On the package
        this is a private static method and I dont have access to it. So I copied
        :(
        I did a modification tho, so this can be an idempotent mapping:
        endenai: format -> openai
        openai: format -> openai

        Format messages into a format accepted by OpenAI.

        Args:
            messages (List[ChatMessageDataClass]): List of messages to be formatted.

        Returns:
            List[Dict[str, str]]: Transformed messages in OpenAI accepted format.

        >>> Accepted format:
            [
                {
                    "role": <role>,
                    "content": [
                        {
                            "type": "text",
                            "text": <text_content>
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": <image_url>}
                        }
                    ]
                }
            ]

        """
        try:
            transformed_messages = []
            for message in messages:
                if message["role"] == "user":
                    transformed_message = {"role": message["role"], "content": []}
                    for item in message["content"]:
                        if item["type"] == "text":
                            if "content" in item.keys():
                                transformed_message["content"].append(
                                    {
                                        "type": "text",
                                        "text": item.get("content").get("text"),
                                    }
                                )
                            else:
                                transformed_message["content"].append(
                                    {"type": "text", "text": item.get("text")}
                                )
                        elif item["type"] == "media_url":
                            media_url = item["content"]["media_url"]
                            media_type = item["content"]["media_type"]
                            response = httpx.get(media_url)
                            data = base64.b64encode(response.content).decode("utf-8")
                            media_data_url = f"data:{media_type};base64,{data}"
                            transformed_message["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": media_data_url},
                                }
                            )
                        elif item["type"] == "image_url":
                            transformed_message["content"].append(item)
                        else:
                            b64_data = item.get("content").get("media_base64")
                            b64_type = item.get("content").get("media_type")
                            media_data_url = f"data:{b64_type};base64,{b64_data}"
                            transformed_message["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": media_data_url},
                                }
                            )
                elif isinstance(message.get("content"), str):
                    transformed_message = {
                        "role": message["role"],
                        "content": message.get("content"),
                    }
                else:
                    transformed_message = {
                        "role": message["role"],
                        "content": message.get("content")[0].get("content").get("text"),
                    }

                transformed_messages.append(transformed_message)

            if system_prompt:
                transformed_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

            return transformed_messages
        except Exception as e:
            raise e

    @staticmethod
    def convert_tools_llmengine(tools: Optional[List[dict]]):
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            # if the tool is google_search_retrieval...
            if tool.get("name") == "google_search_retrieval":
                # ...then we need to convert it to a google_search tool
                openai_tools.append(tool["parameters"])
            else:
                openai_tools.append({"type": "function", "function": tool})
        return openai_tools

    @staticmethod
    def format_classification_examples(examples: List[List[str]]):
        if not examples:
            return ""
        formated_texts = ""
        for i, item in enumerate(examples, start=1):
            formated_texts += f"""{i}. '{item}' """
        text = f"""
                ======
                Example Classification:
                {formated_texts}"""
        return text

    @staticmethod
    def format_ner_examples(examples):
        """Format examples into a clear, readable format for the prompt."""
        if not examples:
            return ""

        formatted_examples = []
        for idx, example in enumerate(examples, 1):
            formatted_entities = "\n".join(
                [
                    f"    - Entity: {entity['entity']}"
                    f"\n      Category: {entity['category']}"
                    for entity in example["entities"]
                ]
            )

            formatted_example = (
                f"Example {idx}:\n"
                f"  Text: {example['text']}\n"
                f"  Entities:\n{formatted_entities}"
            )
            formatted_examples.append(formatted_example)

        return "\n\n".join(formatted_examples)
