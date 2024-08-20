import base64
import os
import json
from io import BytesIO
from json import JSONDecodeError
from typing import Sequence, Literal, Optional
from time import sleep

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

    def image__generation(
        self,
        text: str,
        resolution: Literal["256x256", "512x512", "1024x1024"],
        num_images: int = 1,
        model: Optional[str] = None,
    ) -> ResponseType[ImageGenerationDataClass]:
        self.check_content_moderation(text=text)
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

    def __assistant_image(
        self, name, instruction, message_text, example_file, input_file, dataclass
    ):

        file = self.client.files.create(file=open(input_file, "rb"), purpose="vision")

        with open(os.path.join(os.path.dirname(__file__), example_file), "r") as f:
            output_response = json.load(f)["standardized_response"]

        assistant = self.client.beta.assistants.create(
            response_format={"type": "json_object"},
            model="gpt-4o",
            name=name,
            instructions="{} You return a json output shaped like the following with the exact same structure and the exact same keys but the values would change : \n {} \n\n You should follow this pydantic dataclass schema {}".format(
                instruction, output_response, dataclass.schema()
            ),
        )
        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message_text,
                        },
                        {
                            "type": "image_file",
                            "image_file": {"file_id": file.id},
                        },
                    ],
                }
            ]
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        while run.status != "completed":
            sleep(1)

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        usage = run.to_dict()["usage"]
        original_response = messages.to_dict()
        original_response["usage"] = usage

        try:
            standardized_response = json.loads(
                json.loads(messages.data[0].content[0].json())["text"]["value"]
            )
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return original_response, standardized_response

    def image__explicit_content(
        self,
        file: str,
        file_url: str = "",
    ) -> ResponseType[ExplicitContentDataClass]:
        original_response, result = self.__assistant_image(
            name="Image Explicit Content Analysis",
            instruction="You are an Explicit Image Detection model.",
            message_text="Analys this image :",
            example_file="outputs/image/explicit_content_output.json",
            input_file=file,
            dataclass=ExplicitContentDataClass,
        )

        return ResponseType[ExplicitContentDataClass](
            original_response=original_response,
            standardized_response=result,
        )

    def image__logo_detection(
        self, file: str, file_url: str = "", model: str = None
    ) -> ResponseType[LogoDetectionDataClass]:
        original_response, result = self.__assistant_image(
            name="logo detection",
            instruction="You are a Logo Detection model. You get an image input and return logos detected inside it. If no logo is detected the items list should be empty",
            message_text="Analys this image :",
            example_file="outputs/image/logo_detection_output.json",
            input_file=file,
            dataclass=LogoDetectionDataClass,
        )

        return ResponseType[LogoDetectionDataClass](
            original_response=original_response,
            standardized_response=result,
        )
