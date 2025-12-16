from typing import Optional, Union, Literal
from litellm import completion_cost
from litellm.types.utils import ModelResponse


def calculate_cost(
    completion_response: Union[ModelResponse, dict],
    model: str,
    call_type: Literal[
        "completion",
        "embedding",
        "image_generation",
        "moderation",
        "acompletion",
        "aembedding",
        "aimage_generation",
        "amoderation",
        "arerank",
    ] = "completion",
    input_cost_per_token: Optional[float] = None,
    output_cost_per_token: Optional[float] = None,
) -> float:
    """
    Calculate the cost of a completion response using litellm's completion_cost.

    Args:
        completion_response: The response from litellm completion/embedding/etc.
            Can be a ModelResponse object or a dict.
        call_type: The type of API call that generated the response.
        input_cost_per_token: Custom cost per input token (optional).
        output_cost_per_token: Custom cost per output token (optional).

    Returns:
        The calculated cost in USD as a float.
    """
    cost_calc_params = {
        "completion_response": completion_response,
        "call_type": call_type,
    }

    if input_cost_per_token is not None and output_cost_per_token is not None:
        cost_calc_params["custom_cost_per_token"] = {
            "input_cost_per_token": input_cost_per_token,
            "output_cost_per_token": output_cost_per_token,
        }

    return completion_cost(**cost_calc_params, model=model)
