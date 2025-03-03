from typing import Literal, Optional
from pydantic import BaseModel, Field


class LiteLLMConfig(BaseModel):
    litellm_provider: str
    mode: Literal[
        "completion",
        "chat",
        "embedding",
        "image_generation",
        "audio_transcription",
        "audio_speech",
    ] = "completion"
    max_tokens: int
    max_input_tokens: Optional[int] = 0
    max_output_tokens: Optional[int] = 0
    input_cost_per_token: Optional[float]
    output_cost_per_token: Optional[float]
    supports_function_calling: Optional[bool] = False
    supports_parallel_function_calling: Optional[bool] = False
    supports_vision: Optional[bool] = False
    supports_audio_input: Optional[bool] = False
    supports_audio_output: Optional[bool] = False
    supports_prompt_caching: Optional[bool] = False
    supports_response_schema: Optional[bool] = False
    supports_system_messages: Optional[bool] = False
    deprecation_date: Optional[str] = None  # This is an useful information from LiteLLM
    source: Optional[str] = None


class LiteLLMModel(BaseModel):
    model_name: str
    model_configuration: LiteLLMConfig = Field(..., alias="model_configuration")
