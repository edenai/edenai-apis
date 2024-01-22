import base64
from io import BytesIO
from json import JSONDecodeError
from typing import Sequence, Literal, Optional
import requests

import openai

from edenai_apis.features import ImageInterface
from edenai_apis.features.image.generation import (
    GenerationDataClass as ImageGenerationDataClass,
    GeneratedImageDataClass,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3
from .helpers import (
    get_openapi_response,
)
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
            file : str,
            num_images : int = 1, 
            resolution : Literal["256x256", "512x512", "1024x1024"] = "512x512"
            ) :
        
        """with open(file, "rb") as fstream:
            

            url = f"{self.url}/images/variations"
        
            self.headers['Content-type'] = 'multipart/form-data'
            payload = {
                'image' : fstream.read()
                "model" : "dall-e-2",
                "n" : num_images,
                "size" : resolution,
            }"""

        response = openai.Image.create_variation(
            image = open(file, 'rb'),
            n = num_images,
            model = 'dall-e-2',
            size = resolution,
        )
    
        original_response = response
        print(response)