from typing import Dict, List, Optional

from edenai_apis.features import ProviderInterface, ImageInterface
from edenai_apis.features.image import (
    GenerationFineTuningCreateProjectAsyncDataClass,
    GenerationFineTuningGenerateImageAsyncDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncBaseResponseType,
)


class AstriaApi(ProviderInterface, ImageInterface):
    provider_name = "astria"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.url = "https://api.astria.ai/"

        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["api_key"]
        self.headers = {"authorization": f"Bearer {self.api_key}"}

    def image__generation_fine_tuning__create_project_async__launch_job(
        self,
        name: str,
        description: str,
        files: List[str],
        files_url: List[str] = [],
        base_project_id: Optional[int] = None,
    ) -> AsyncLaunchJobResponseType:
        raise NotImplementedError

    def image__generation_fine_tuning__create_project_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[GenerationFineTuningCreateProjectAsyncDataClass]:
        raise NotImplementedError

    def image__generation_fine_tuning__generate_image_async__launch_job(
        self,
        project_id: str,
        prompt: str,
        negative_prompt: Optional[str] = "",
        num_images: Optional[int] = 1,
    ) -> AsyncLaunchJobResponseType:
        raise NotImplementedError

    def image__generation_fine_tuning__generate_image_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[GenerationFineTuningGenerateImageAsyncDataClass]:
        raise NotImplementedError
