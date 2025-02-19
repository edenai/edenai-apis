import os
import logging
from typing import List, Optional, Union
from llmengine.types.response_types import (
    CustomStreamWrapperModel,
    EmbeddingResponseModel,
    ResponseModel,
)
from llmengine.exceptions.llm_engine_exceptions import CompletionClientError

logger = logging.getLogger(__name__)


class CompletionClient:

    def __init__(
        self,
        provider_name: str = None,
        model_name: Optional[str] = None,
        provider_config: dict = {},
    ) -> None:
        self.model_name = model_name
        self.provider_name = provider_name
        self.provider_config = provider_config
        self.model_capabilities = []
        self.call_params = {}

    def completion(
        self, messages: List = [], model: str = None, **kwargs
    ) -> Union[ResponseModel, CustomStreamWrapperModel]:
        raise CompletionClientError(
            "Not implemented. Please implement this method in a subclass."
        )

    def embedding(
        self, text: str, model: str = None, **kwargs
    ) -> EmbeddingResponseModel:
        raise CompletionClientError(
            "Not implemented. Please implement this method in a subclass."
        )

    def moderation(self, input: str, api_key: Optional[str] = None):
        raise CompletionClientError(
            "Not implemented. Please implement this method in a subclass."
        )

    def image_generation(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ):
        raise CompletionClientError(
            "Not implemented. Please implement this method in a subclass."
        )

    def _get_unsupported_params(self, **call_configuration) -> List[str]:
        """
        Checks whether the client is capable of handling the provided configuration.
        Returns a list with the invalid given parameters
        """
        if len(self.model_capabilities) == 0:
            logger.warning(
                "Model capabilities are not set. Assuming the client is capable of handling the provided configuration."
            )
            return []
        # Ignore messages and model
        IGNORED_KEYS = [
            "messages",
            "model",
            "api_key",
            "aws_region_name",
            "aws_access_key_id",
            "aws_secret_access_key",
        ]
        filtered_call_configuration = {
            k: v for k, v in call_configuration.items() if k not in IGNORED_KEYS
        }
        invalid_parameters = set(filtered_call_configuration.keys()) - set(
            self.model_capabilities
        )
        return list(invalid_parameters)

    def __str__(self) -> str:
        return str(f"model_name: {self.model_name} provider_name: {self.provider_name}")

    def _configure_bedrock(self, auth_args):
        auth_info = {
            "aws_access_key_id": auth_args["aws_access_key_id"],
            "aws_secret_access_key": auth_args["aws_secret_access_key"],
            "aws_region_name": auth_args["region_name"],
        }
        return auth_info

    def _configure_google(self, auth_args):
        os.environ["GEMINI_API_KEY"] = auth_args["genai_api_key"]

    @staticmethod
    def std_completion():
        raise CompletionClientError(
            "Not implemented. Please implement this method in a subclass."
        )
