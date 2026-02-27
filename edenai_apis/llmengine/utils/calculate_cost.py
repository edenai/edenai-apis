from typing import Optional, Union
from litellm import completion_cost
from litellm.types.utils import ModelResponse, CallTypesLiteral


def calculate_cost(
    completion_response: Union[ModelResponse, dict],
    model: str,
    input_cost_per_token: Optional[float] = None,
    output_cost_per_token: Optional[float] = None,
    call_type: CallTypesLiteral = "acompletion",
) -> float:
    """
    Calculate the cost of a completion response using litellm's completion_cost.

    Args:
        completion_response: The response from litellm completion/embedding/etc.
            Can be a ModelResponse object or a dict.
        model: Model name for pricing lookup (e.g., 'openai/gpt-4'). Required
            because streaming responses may lack the provider prefix needed
            for pricing lookup in litellm.model_cost.
        call_type: The type of API call that generated the response.
        input_cost_per_token: Custom cost per input token (optional).
        output_cost_per_token: Custom cost per output token (optional).

    Returns:
        The calculated cost in USD as a float.
    """
    cost_calc_params = {
        "completion_response": completion_response,
        "call_type": call_type,
        "model": model,
    }

    if input_cost_per_token is not None and output_cost_per_token is not None:
        cost_calc_params["custom_cost_per_token"] = {
            "input_cost_per_token": input_cost_per_token,
            "output_cost_per_token": output_cost_per_token,
        }

    return completion_cost(**cost_calc_params)
