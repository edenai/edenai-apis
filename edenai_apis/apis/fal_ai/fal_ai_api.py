from typing import Dict, Literal, Optional

from edenai_apis.features.image.generation.generation_dataclass import (
    GenerationDataClass,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.llmengine.utils.moderation import async_moderate
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import ResponseType


class FalAiApi(ProviderInterface, ImageInterface):
    provider_name = "fal_ai"

    def __init__(self, api_keys: Dict = {}):
        # transformed_provider_name = self.provider_name.replace("-", "_")
        # api_settings = load_provider(
        #     ProviderDataEnum.KEY,
        #     provider_name=transformed_provider_name,
        #     api_keys=api_keys,
        # )
        self.api_key = None#api_settings.get("api_key")

        self.llm_client = LLMEngine(
            provider_name=self.provider_name,
            provider_config={
                "api_key": self.api_key,
            },
        )

        self.moderation_flag = True

    @async_moderate
    async def image__ageneration(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[GenerationDataClass]:
        response = await self.llm_client.aimage_generation(
            prompt=text,
            resolution=resolution,
            n=num_images,
            model=model,
        )
        return response
