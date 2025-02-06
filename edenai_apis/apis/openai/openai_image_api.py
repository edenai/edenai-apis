import base64
from io import BytesIO
from typing import Sequence, Literal, Optional

from openai import APIError


import requests
from edenai_apis.features import ImageInterface
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
)
from edenai_apis.features.image.generation import (
    GenerationDataClass as ImageGenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoDetectionDataClass,
)
from edenai_apis.features.image.variation import (
    VariationDataClass,
    VariationImageDataClass,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from .helpers import get_openapi_response
from ...features.image.question_answer import QuestionAnswerDataClass
from ...utils.exception import ProviderException


class OpenaiImageApi(ImageInterface):

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
    ) -> ResponseType[ImageGenerationDataClass]:
        response = self.llm_client.image_generation(
            prompt=text, resolution=resolution, n=num_images, model=model
        )
        return response

    def image__question_answer(
        self,
        file: str,
        temperature: float,
        max_tokens: int,
        file_url: str = "",
        model: Optional[str] = None,
        question: Optional[str] = None,
    ) -> ResponseType[QuestionAnswerDataClass]:
        response = self.llm_client.image_qa(
            file=file,
            temperature=temperature,
            max_tokens=max_tokens,
            file_url=file_url,
            model=model,
            question=question,
        )
        return response

    def image__variation(
        self,
        file: str,
        prompt: Optional[str],
        num_images: Optional[int] = 1,
        resolution: Literal["256x256", "512x512", "1024x1024"] = "512x512",
        temperature: Optional[float] = 0.3,
        model: Optional[str] = None,
        file_url: str = "",
    ) -> ResponseType[VariationDataClass]:
        try:
            with open(file, "rb") as file_:
                response = self.client.images.create_variation(
                    image=file_,
                    n=num_images,
                    model=model,
                    size=resolution,
                    response_format="b64_json",
                )
        except APIError as error:
            raise ProviderException(message=error.user_message, code=error.code)

        original_response = response
        generations: Sequence[VariationImageDataClass] = []
        for generated_image in original_response.data:
            image_b64 = generated_image.b64_json
            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(image_content, ".png", USER_PROCESS)
            generations.append(
                VariationImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )

        return ResponseType[VariationDataClass](
            original_response=original_response.to_dict(),
            standardized_response=VariationDataClass(items=generations),
        )

    def image__explicit_content(
        self, file: str, file_url: str = "", model: Optional[str] = None
    ) -> ResponseType[ExplicitContentDataClass]:
        response = self.llm_client.image_moderation(
            file=file, file_url=file_url, model=model
        )
        return response

    def image__logo_detection(
        self, file: str, file_url: str = "", model: str = None
    ) -> ResponseType[LogoDetectionDataClass]:
        response = self.llm_client.logo_detection(
            file=file, file_url=file_url, model=model
        )
        return response
