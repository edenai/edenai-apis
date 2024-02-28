from typing import Dict, Sequence, Optional, Literal, Any

import requests
from aleph_alpha_client import (
    Client,
    Prompt,
    SemanticEmbeddingRequest,
    Image,
    SemanticRepresentation,
    CompletionRequest,
    Text,
)

from edenai_apis.features import ProviderInterface, TextInterface, ImageInterface
from edenai_apis.features.image.embeddings import (
    EmbeddingsDataClass,
    EmbeddingDataClass,
)
from edenai_apis.features.image.question_answer import QuestionAnswerDataClass
from edenai_apis.features.multimodal import MultimodalInterface
from edenai_apis.features.multimodal.embeddings import (
    EmbeddingsDataClass as MultimodalEmbeddingsDataClass,
    EmbeddingModel,
)
from edenai_apis.features.text import SummarizeDataClass
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class AlephAlphaApi(
    ProviderInterface, TextInterface, ImageInterface, MultimodalInterface
):
    provider_name = "alephalpha"

    def __init__(self, api_keys: Optional[Dict[str, Any]] = None):
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys or {}
        )
        self.api_key = self.api_settings["api_key"]
        self.url_basic = "https://api.aleph-alpha.com"
        self.url_summarise = "https://api.aleph-alpha.com/summarize"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def __construct_prompt(
        text: Optional[Text],
        image: Optional[Image],
    ) -> Prompt:
        if not text:
            return Prompt([image])
        if not image:
            return Prompt([text])
        return Prompt([text, image])

    def text__summarize(
        self,
        text: str,
        output_sentences: int,
        language: str,
        model: str,
    ) -> ResponseType[SummarizeDataClass]:
        payload = {"model": model, "document": {"text": text}}
        response = requests.post(
            url=self.url_summarise, headers=self.headers, json=payload
        )
        if response.status_code != 200:
            raise ProviderException(response.text, code=response.status_code)
        original_response = response.json()
        standardized_response = SummarizeDataClass(
            result=original_response.get("summary", {})
        )
        return ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def image__embeddings(
        self,
        file: str,
        model: str,
        representation: str,
        file_url: str = "",
    ) -> ResponseType[EmbeddingsDataClass]:
        if representation == "symmetric":
            representation_client = SemanticRepresentation.Symmetric
        elif representation == "document":
            representation_client = SemanticRepresentation.Document
        else:
            representation_client = SemanticRepresentation.Query
        client = Client(self.api_key)
        prompt = Prompt.from_image(Image.from_file(file))
        request = SemanticEmbeddingRequest(
            prompt=prompt, representation=representation_client
        )
        try:
            response = client.semantic_embed(request=request, model=model)
        except Exception as exc:
            raise ProviderException(message=str(exc)) from exc

        original_response = response.__dict__
        items: Sequence[EmbeddingDataClass] = [
            EmbeddingDataClass(embedding=response.embedding)
        ]
        standardized_response = EmbeddingsDataClass(items=items)
        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
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
        client = Client(self.api_key)
        if question:
            prompts = Prompt([Text.from_text(question), Image.from_file(file)])
        else:
            prompts = Prompt([Image.from_file(file)])
        request = CompletionRequest(
            prompt=prompts,
            maximum_tokens=max_tokens,
            temperature=temperature,
            tokens=True,
        )
        try:
            response = client.complete(request=request, model=model)
        except Exception as error:
            raise ProviderException(str(error)) from error
        original_response = response._asdict()
        answers = []
        for answer in response.completions:
            answers.append(answer.completion)
        standardized_response = QuestionAnswerDataClass(answers=answers)
        return ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )

    def multimodal__embeddings(
        self,
        model: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        dimension: Literal["xs", "s", "m", "xl"] = "xl",
    ) -> ResponseType[MultimodalEmbeddingsDataClass]:
        client = Client(self.api_key)
        parsed_inputs = {
            "text": text,
            "image": image,
            "image_url": image_url,
            "video": video,
            "video_url": video_url,
        }

        request_text = (
            Text.from_text(parsed_inputs["text"]) if parsed_inputs["text"] else None
        )
        request_image = image = (
            Image.from_url(parsed_inputs["image_url"])
            if parsed_inputs["image_url"]
            else Image.from_file(parsed_inputs["image"])
            if parsed_inputs["image"]
            else None
        )

        request = SemanticEmbeddingRequest(
            prompt=AlephAlphaApi.__construct_prompt(request_text, request_image),
            representation=SemanticRepresentation.Symmetric,
        )
        try:
            response = client.semantic_embed(request=request, model=model)
        except Exception as exc:
            raise ProviderException(message=str(exc)) from exc

        original_response = response.__dict__
        standardized_response = MultimodalEmbeddingsDataClass(
            items=[
                EmbeddingModel(
                    text_embedding=response.embedding if request_text else [],
                    image_embedding=response.embedding if request_image else [],
                )
            ]
        )
        return ResponseType[MultimodalEmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
