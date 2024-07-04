import base64
from io import BytesIO
from json import JSONDecodeError
from typing import Sequence, Literal, Optional

from openai import OpenAI, APIError


import requests
import mimetypes

from edenai_apis.features import ImageInterface
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
    CategoryType,
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
from .tools import OpenAIFunctionTools
from .helpers import get_openapi_response
from ...features.image.question_answer import QuestionAnswerDataClass
from ...utils.exception import ProviderException


class OpenaiImageApi(ImageInterface):

    @staticmethod
    def __image_request(
        file: str,
        model: str,
        tools: list[dict],
        tool_choice: dict,
        system_prompt: Optional[str] = None,
    ):
        """
        Sends an image file to the OpenAI API for processing, such as logo detection or explicit content detection.
        Args:
            file (str): The path to the image file to be processed.
            model (str): The name of the OpenAI model to be used for processing.
            tools (list): list of tools to use.
            tool_choice (dict) : choice of used tool
        """
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as fstream:
            base64_data = base64.b64encode(fstream.read()).decode("utf-8")
            media_data_url = f"data:{mime_type};base64,{base64_data}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": media_data_url}}
                ],
            },
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        payload = {
            "messages": messages,
            "model": model,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as exc:
            raise ProviderException(str(exc)) from exc

        return response

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
            response = self.client.images.generate(
                image=open(file, "rb"),
                n=num_images,
                model=model,
                size=resolution,
                response_format="b64_json",
            )
        except APIError as error:
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
        self, file: str, file_url: str = "", model: str = None
    ) -> ResponseType[LogoDetectionDataClass]:
        openai_function_tools = OpenAIFunctionTools(
            tool_name="logo_detection",
            tool_description="Detects logos and brands in an image. This function identifies the logos present in the image.",
            dataclass=LogoDetectionDataClass,
        )
        response = self.__image_request(
            file=file,
            model=model,
            tools=openai_function_tools.get_tool(),
            tool_choice=openai_function_tools.get_tool_choice(),
        )
        standardized_response = openai_function_tools.get_response(response=response)

        return ResponseType[LogoDetectionDataClass](
            original_response=response,
            standardized_response=standardized_response,
        )

    def image__explicit_content(
        self,
        file: str,
        file_url: str = "",
    ) -> ResponseType[ExplicitContentDataClass]:

        openai_function_tools = OpenAIFunctionTools(
            tool_name="explicit_content_detection",
            tool_description="Detects Explicit content in an image.",
            dataclass=ExplicitContentDataClass,
        )
        response = self.__image_request(
            file=file,
            model="gpt-4o",
            tools=openai_function_tools.get_tool(),
            tool_choice=openai_function_tools.get_tool_choice(),
        )
        standardized_response = openai_function_tools.get_response(response=response)
        return ResponseType(
            original_response=response,
            standardized_response=standardized_response,
        )
