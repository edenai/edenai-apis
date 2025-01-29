from typing import List, Optional, Dict
import json


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
            tool_call = get_tool_call_from_history_by_id(tool["id"], previous_history)
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


def convert_tools_to_openai(tools: Optional[List[dict]]):
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({"type": "function", "function": tool})
    return openai_tools
