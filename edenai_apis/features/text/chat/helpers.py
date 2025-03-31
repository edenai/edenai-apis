import itertools
from typing import Dict, List

from edenai_apis.utils.exception import ProviderException


def get_tool_call_from_history_by_id(id: str, previous_history: List[Dict]):
    """
    Check all tool_calls of all messages.
    Returns the tool call with the given id.
    """
    tool_calls = itertools.chain.from_iterable(
        [msg["tool_calls"] for msg in previous_history if msg["tool_calls"]]
    )
    tool_call = list(filter(lambda tool: tool["id"] == id, tool_calls))
    if not tool_call:
        raise ProviderException(
            f"The id {id} is not correct. "
            "Please make sure to add the assistant message containing "
            "tool calls to history, and check tool calls ids."
        )
    return tool_call[0]
