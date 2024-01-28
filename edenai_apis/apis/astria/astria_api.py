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

import requests

def load_image(file_path):
    with open(file_path, "rb") as f:
        return f.read()

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
            title: str,
            class_name: str,
            files: List[str] = [],
            files_url: List[str] = [],
            base_tune_id: Optional[int] = None,
    ) -> AsyncLaunchJobResponseType:
        data = {
            "tune[title]": title,
            "tune[name]": class_name,
            "tune[base_tune_id]": 690204,
            # "tune[callback]": 'https://optional-callback-url.com/to-your-service-when-ready?prompt_id=1'
        }
        for image in files:
            image_data = load_image(image)  # Assuming image is a file path
            files.append(("tune[images][]", image_data))
        for image_url in files_url:
            files.append(("tune[image_urls][]", image_url))

        response = requests.post(f"{self.url}tunes", data=data, files=files, headers=self.headers)
        response.raise_for_status()
        return AsyncLaunchJobResponseType(provider_job_id=response.json()["id"])

    def image__generation_fine_tuning__create_project_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[GenerationFineTuningCreateProjectAsyncDataClass]:
        response = requests.get(f"{self.url}tunes/{provider_job_id}", headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return AsyncBaseResponseType(
            status="succeeded" if data['trained_at'] else "pending",
            provider_job_id=provider_job_id,
            original_response=data,
            standardized_response=GenerationFineTuningCreateProjectAsyncDataClass(**data),
        )

    def image__generation_fine_tuning__generate_image_async__launch_job(
            self,
            project_id: str,
            prompt: str,
            negative_prompt: Optional[str] = "",
            num_images: Optional[int] = 1,
            input_image: Optional[str] = None,
    ) -> AsyncLaunchJobResponseType:
        data = {
            'prompt[text]': prompt,
            'prompt[negative_prompt]': negative_prompt,
            'prompt[num_images]': num_images,
            'prompt[face_swap]': True,
            'prompt[inpaint_faces]': False,
            'prompt[super_resolution]': True,
            'prompt[face_correct]': False,
            # 'prompt[callback]': 'https://optional-callback-url.com/to-your-service-when-ready?prompt_id=1'
        }
        files = []
        if input_image:
            files.append((f"tune[prompts_attributes][{i}][input_image]", load_image(input_image)))

        response = requests.post(f"{self.url}/tunes/{project_id}", headers=self.headers, data=data)
        response.raise_for_status()
        return AsyncLaunchJobResponseType(provider_job_id=response.json()["id"])

    def image__generation_fine_tuning__generate_image_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[GenerationFineTuningGenerateImageAsyncDataClass]:
        response = requests.get(f"{self.url}tunes/{provider_job_id}", headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return AsyncBaseResponseType(
            status="succeeded" if data['trained_at'] else "pending",
            provider_job_id=provider_job_id,
            original_response=data,
            standardized_response=GenerationFineTuningGenerateImageAsyncDataClass(**data),
        )
