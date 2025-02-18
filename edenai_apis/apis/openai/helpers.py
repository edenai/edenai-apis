import json
from enum import Enum
from typing import List, Optional

from requests import Response

from edenai_apis.utils.exception import ProviderException
from .prompts_guidelines import (
    anthropic_prompt_guidelines,
    cohere_prompt_guideines,
    google_prompt_guidelines,
    general_prompt_guidelines,
    perplexityai_prompt_guidelines,
)


class OpenAIErrorCode(Enum):
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


def get_openapi_response(response: Response):
    """
    This function takes a requests.Response as input and return it's response.json()
    raises a ProviderException if the response contains an error.
    """
    try:
        original_response = response.json()
        if "error" in original_response or response.status_code >= 400:
            message_error = original_response["error"]["message"]
            raise ProviderException(message_error, code=response.status_code)
        return original_response
    except Exception:
        raise ProviderException(response.text, code=response.status_code)


prompt_optimization_missing_information = (
    lambda user_description: f"""
You are a Prompt Optimizer for LLMs, you take a description in input and generate a prompt from it.

The user's project description is delimited by triple back-ticks.

You should to tell him to provide you what any missing information about his project would guide you towards giving him a better prompt.

If the description is clear don't provide any missing information.

The User's description :

```{user_description}```

missing information : 
"""
)


def construct_prompt_optimization_instruction(text: str, target_provider: str):
    """
    Constructs prompt optimization instructions based on the target provider.

    Args:
        text (str): The input text for which prompt optimization instructions are needed.
        target_provider (str): The target provider for which prompt optimization instructions are requested.

    Returns:
        prompt: A str containing the prompt optimization instructions for the specified target provider.
    """
    prompt = {
        "google": google_prompt_guidelines(text),
        "cohere": cohere_prompt_guideines(text),
        "openai": general_prompt_guidelines(text, "OpenAI", "GPT"),
        "mistral": general_prompt_guidelines(text, "Mistral", "open"),
        "meta": general_prompt_guidelines(text, "Meta", "Llama"),
        "anthropic": anthropic_prompt_guidelines(text),
        "perplexityai": perplexityai_prompt_guidelines(text),
    }

    # Check if the target provider is supported, if not raise error
    if target_provider not in prompt:
        raise ProviderException(f"Unsupported target provider: {target_provider}")

    return prompt[target_provider]


def convert_tts_audio_rate(audio_rate: int) -> float:
    """
    Convert TTS audio rate from the range [-100, 100] to [0.25, 4.0].

    Parameters:
    - audio_rate (int): The input audio rate in the range [-100, 100].

    Returns:
    - float: The audio rate in the range [0.25, 4.0].

    """
    if audio_rate >= -100 and audio_rate <= 0:
        return ((audio_rate - (-100)) / (0 - (-100))) * (1 - 0.25) + 0.25
    else:
        return ((audio_rate - 0) / (100 - 0)) * (4 - 1) + 1


def finish_unterminated_json(json_string: str, end_brackets: str) -> str:
    """
    take a cut json_string and try to terminate it

    Arguments:
      json_string(str): JSON string to terminate
      end_brackets(str): string representing the ending brackets. eg: '}]'

    Returns:
      str: valid json string

    Raise:
      json.JSONDecodeERror: if couldn't terminate string

    Example:
      >>> finish_unterminated_json('{"data": {"place": "Italy"')
      >>> '{"data": {"place": "Italy"}}
    """
    if not json_string:
        raise json.JSONDecodeError("JSON string couldn't be parsed or finished")

    try:
        new_json_string = json_string + end_brackets
        json.loads(new_json_string)
        return new_json_string
    except json.JSONDecodeError:
        json_string = json_string[:-1]
        return finish_unterminated_json(json_string, end_brackets)


def convert_tools_to_openai(tools: Optional[List[dict]]):
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({"type": "function", "function": tool})
    return openai_tools


def convert_tool_results_to_openai_tool_calls(tools_results: List[dict]):
    result = []
    for tool in tools_results:
        result.append(
            {
                "id": tool["call"]["id"],
                "type": "function",
                "function": {
                    "name": tool["call"]["name"],
                    "arguments": tool["call"]["arguments"],
                },
            }
        )
    return result
