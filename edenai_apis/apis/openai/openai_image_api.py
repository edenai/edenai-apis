import base64
import json
import mimetypes
from io import BytesIO
from json import JSONDecodeError
from typing import Sequence, Literal, Optional, List
from pydantic_core._pydantic_core import ValidationError

import openai
import requests

from edenai_apis.features import ImageInterface
from edenai_apis.features.image.generation import (
    GenerationDataClass as ImageGenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoDetectionDataClass,
    LogoItem,
)
from edenai_apis.features.image.variation import (
    VariationDataClass,
    VariationImageDataClass,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from .helpers import (
    get_openapi_response,
)
from edenai_apis.apis.anthropic.prompts import LOGO_DETECTION_SYSTEM_PROMPT
from ...features.image.question_answer import QuestionAnswerDataClass
from ...utils.exception import ProviderException


class OpenaiImageApi(ImageInterface):

    @staticmethod
    def __image_request(file: str, model: str, system_prompt: str):
        """
        Sends an image file to the OpenAI API for processing, such as logo detection or explicit content detection.

        Args:
            file (str): The path to the image file to be processed.
            model (str): The name of the OpenAI model to be used for processing.
            system_prompt (str): The system prompt to be used for the model.
        """
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as fstream:
            base64_data = base64.b64encode(fstream.read()).decode("utf-8")
            media_data_url = f"data:{mime_type};base64,{base64_data}"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": media_data_url}}
                ],
            },
        ]
        payload = {
            "messages": messages,
            "model": model,
            "response_format": {"type": "json_object"},
        }

        try:
            response = openai.ChatCompletion.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

        try:
            items = json.loads(response["choices"][0]["message"]["content"])
        except (KeyError, json.JSONDecodeError, ValidationError) as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        return response, items

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
    ) -> ResponseType[ImageGenerationDataClass]:
        url = f"{self.url}/images/generations"
        payload = {
            "prompt": text,
            "model": model,
            "n": num_images,
            "size": resolution,
            "response_format": "b64_json",
        }
        response = requests.post(url, json=payload, headers=self.headers)
        original_response = get_openapi_response(response)

        generations: Sequence[GeneratedImageDataClass] = []
        for generated_image in original_response.get("data"):
            image_b64 = generated_image.get("b64_json")

            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(image_content, ".png", USER_PROCESS)
            generations.append(
                GeneratedImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )

        return ResponseType[ImageGenerationDataClass](
            original_response=original_response,
            standardized_response=ImageGenerationDataClass(items=generations),
        )

    def image__question_answer(
        self,
        file: str,
        temperature: float,
        max_tokens: int,
        file_url: str = "",
        model: Optional[str] = None,
        question: Optional[str] = None,
    ) -> ResponseType[QuestionAnswerDataClass]:
        with open(file, "rb") as fstream:
            file_content = fstream.read()
            file_b64 = base64.b64encode(file_content).decode("utf-8")

            url = f"{self.url}/chat/completions"
            payload = {
                "model": "gpt-4-vision-preview" or model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question or "Describe the following image",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{file_b64}"
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = requests.post(url, json=payload, headers=self.headers)

            if response.status_code >= 500:
                raise ProviderException(
                    f"OpenAI API is not available. Status code: {response.status_code}"
                )

            if response.status_code != 200:
                raise ProviderException(
                    message=response.text, code=response.status_code
                )

            try:
                original_response = response.json()
            except JSONDecodeError as exc:
                raise ProviderException(
                    message="Invalid JSON response", code=response.status_code
                ) from exc

            standardized_response = QuestionAnswerDataClass(
                answers=[original_response["choices"][0]["message"]["content"]]
            )

            return ResponseType[QuestionAnswerDataClass](
                original_response=original_response,
                standardized_response=standardized_response,
            )

    def image__variation(
        self,
        file: str,
        prompt: Optional[str] = "",
        num_images: Optional[int] = 1,
        resolution: Literal["256x256", "512x512", "1024x1024"] = "512x512",
        temperature: Optional[int] = 0.3,
        model: Optional[str] = None,
        file_url: str = "",
    ) -> ResponseType[VariationDataClass]:
        try:
            response = openai.Image.create_variation(
                image=open(file, "rb"),
                n=num_images,
                model=model,
                size=resolution,
                response_format="b64_json",
            )
        except openai.OpenAIError as error:
            raise ProviderException(message=error.user_message, code=error.code)

        original_response = response
        generations: Sequence[VariationImageDataClass] = []
        for generated_image in original_response.get("data"):
            image_b64 = generated_image.get("b64_json")

            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(image_content, ".png", USER_PROCESS)
            generations.append(
                VariationImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )

        return ResponseType[VariationDataClass](
            original_response=original_response,
            standardized_response=VariationDataClass(items=generations),
        )

    def image__logo_detection(
        self, file: str, file_url: str = "", model: Optional[str] = None
    ) -> ResponseType[LogoDetectionDataClass]:
        original_response, logos = self.__image_request(
            file=file, model=model, system_prompt=LOGO_DETECTION_SYSTEM_PROMPT
        )
        items: List[LogoItem] = [
            LogoItem(description=logo, bounding_poly=None, score=None)
            for logo in logos.get("items", [])
        ]

        return ResponseType[LogoDetectionDataClass](
            original_response=original_response,
            standardized_response=LogoDetectionDataClass(items=items),
        )
