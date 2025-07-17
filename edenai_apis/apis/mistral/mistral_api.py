import base64
import json
from typing import Dict, List, Literal, Optional, Type, Union
import httpx
from openai import BaseModel
import requests

from edenai_apis.features import ProviderInterface, TextInterface, OcrInterface
from edenai_apis.features.ocr.ocr.ocr_dataclass import OcrDataClass
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import (
    OcrAsyncDataClass,
    Page,
    Line,
    BoundingBox,
)
from edenai_apis.features.text.chat.chat_dataclass import ChatDataClass, StreamChat
from edenai_apis.features.multimodal.chat.chat_dataclass import (
    ChatDataClass as ChatMultimodalDataClass,
    StreamChat as StreamChatMultimodal,
    ChatMessageDataClass as ChatMultimodalMessageDataClass,
)
from edenai_apis.features.text.embeddings import EmbeddingsDataClass
from edenai_apis.features.text.generation.generation_dataclass import (
    GenerationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    ResponseType,
    AsyncResponseType,
)
from edenai_apis.llmengine.llm_engine import LLMEngine
from edenai_apis.features.llm.llm_interface import LlmInterface
from edenai_apis.features.llm.chat.chat_dataclass import ChatDataClass


class MistralApi(ProviderInterface, TextInterface, LlmInterface, OcrInterface):
    provider_name = "mistral"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        self.api_key = self.api_settings["api_key"]
        self.url = "https://api.mistral.ai/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.llm_client = LLMEngine(
            provider_name=self.provider_name, provider_config={"api_key": self.api_key}
        )

    def text__generation(
        self, text: str, temperature: float, max_tokens: int, model: str, **kwargs
    ) -> ResponseType[GenerationDataClass]:
        messages = [
            {
                "role": "system",
                "content": "Act as Text Generator, complete the given texts.",
            },
            {"role": "user", "content": text},
        ]

        payload = {
            "model": f"{self.provider_name}-{model}",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            self.url + "v1/chat/completions", json=payload, headers=self.headers
        )
        try:
            original_response = response.json()
            if "message" in original_response or response.status_code >= 400:
                message_error = original_response["message"]
                raise ProviderException(message_error, code=response.status_code)
        except Exception:
            raise ProviderException(response.text, code=response.status_code)

        generated_text = original_response["choices"][0]["message"]["content"]

        # Calculate number of tokens :
        original_response["usage"]["total_tokens"] = (
            original_response["usage"]["completion_tokens"]
            + original_response["usage"]["prompt_tokens"]
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response=GenerationDataClass(generated_text=generated_text),
        )

    def text__chat(
        self,
        text: str,
        chatbot_global_action: Optional[str] = None,
        previous_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stream: bool = False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        response = self.llm_client.chat(
            text=text,
            chatbot_global_action=chatbot_global_action,
            previous_history=previous_history,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stream=stream,
            available_tools=available_tools,
            tool_choice=tool_choice,
            tool_results=tool_results,
            **kwargs,
        )
        return response

    def text__embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[EmbeddingsDataClass]:
        model = model.split("__")[1] if "__" in model else model
        response = self.llm_client.embeddings(
            texts=texts,
            model=model,
            **kwargs,
        )
        return response

    def multimodal__chat(
        self,
        messages: List[ChatMultimodalMessageDataClass],
        chatbot_global_action: Optional[str],
        temperature: float = 0,
        max_tokens: int = 25,
        model: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        stream: bool = False,
        provider_params: Optional[dict] = None,
        response_format=None,
        **kwargs,
    ) -> ResponseType[Union[ChatMultimodalDataClass, StreamChatMultimodal]]:
        response = self.llm_client.multimodal_chat(
            messages=messages,
            chatbot_global_action=chatbot_global_action,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stop_sequences=stop_sequences,
            top_k=top_k,
            top_p=top_p,
            stream=stream,
            response_format=response_format,
            **kwargs,
        )
        return response

    def llm__chat(
        self,
        messages: List = [],
        model: Optional[str] = None,
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[str] = None,
        stop_sequences: Optional[any] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        modalities: Optional[List[Literal["text", "audio", "image"]]] = None,
        audio: Optional[Dict] = None,
        # openai v1.0+ new params
        response_format: Optional[
            Union[dict, Type[BaseModel]]
        ] = None,  # Structured outputs
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated params by OpenAI -> This should be replaced by tools
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ) -> ChatDataClass:
        response = self.llm_client.completion(
            messages=messages,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response

    async def llm__achat(
        self,
        messages: List = [],
        model: Optional[str] = None,
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[str] = None,
        stop_sequences: Optional[any] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        modalities: Optional[List[Literal["text", "audio", "image"]]] = None,
        audio: Optional[Dict] = None,
        # openai v1.0+ new params
        response_format: Optional[
            Union[dict, Type[BaseModel]]
        ] = None,  # Structured outputs
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # soon to be deprecated params by OpenAI -> This should be replaced by tools
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        drop_invalid_params: bool = True,  # If true, all the invalid parameters will be ignored (dropped) before sending to the model
        user: str | None = None,
        # Optional parameters
        **kwargs,
    ) -> ChatDataClass:
        response = await self.llm_client.acompletion(
            messages=messages,
            model=model,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            drop_invalid_params=drop_invalid_params,
            user=user,
            modalities=modalities,
            audio=audio,
            **kwargs,
        )
        return response

    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        url = "https://api.mistral.ai/v1/ocr"
        if file_url:
            image_data = file_url
        else:
            with open(file, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                image_data = f"data:image/jpeg;base64,{base64_image}"

        document = {"type": "image_url", "image_url": image_data}
        payload = {"model": "mistral-ocr-latest", "document": document}
        response = requests.post(url=url, headers=self.headers, json=payload)
        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message=response.text, code=response.status_code
            ) from exc
        if response.status_code != 200:
            raise ProviderException(
                message=response_data.get("message", response.text),
                code=response.status_code,
            )

        standardized_response = OcrDataClass(text=response_data["pages"][0]["markdown"])

        return ResponseType[OcrDataClass](
            original_response=response_data, standardized_response=standardized_response
        )

    def ocr__ocr_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        url = "https://api.mistral.ai/v1/files"
        with open(file, "rb") as f:
            files = {"file": f}
            data = {"purpose": "ocr"}
            response = requests.post(
                url=url, headers=self.headers, files=files, data=data
            )

        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message=response.text, code=response.status_code
            ) from exc
        if response.status_code != 200:
            raise ProviderException(
                message=response_data.get("message", response.text),
                code=response.status_code,
            )
        return AsyncLaunchJobResponseType(provider_job_id=response_data["id"])

    def ocr__ocr_async__get_job_result(
        self, provider_job_id: str
    ) -> AsyncBaseResponseType[OcrAsyncDataClass]:
        url = f"https://api.mistral.ai/v1/files/{provider_job_id}/url?expiry=24"
        response = requests.get(url=url, headers=self.headers)
        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message=response.text, code=response.status_code
            ) from exc
        if response.status_code != 200:
            raise ProviderException(
                message=response_data.get("message", response.text),
                code=response.status_code,
            )
        file_url = response_data["url"]
        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": file_url,
            },
        }
        url = "https://api.mistral.ai/v1/ocr"
        response = requests.post(url=url, headers=self.headers, json=payload)
        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderException(
                message=response.text, code=response.status_code
            ) from exc
        if response.status_code != 200:
            raise ProviderException(
                message=response_data.get("message", response.text),
                code=response.status_code,
            )
        number_of_pages = response_data["usage_info"]["pages_processed"]
        raw_text = ""
        pages = []
        for page in response_data["pages"]:
            raw_text += page["markdown"]
            markdown_lines = page["markdown"].split("\n")
            lines = []
            for line_text in markdown_lines:
                line = Line(text=line_text, confidence=100)
                lines.append(line)
            pages.append(Page(lines=lines))

        return AsyncResponseType(
            original_response=response_data,
            standardized_response=OcrAsyncDataClass(
                raw_text=raw_text, pages=pages, number_of_pages=number_of_pages
            ),
            provider_job_id=provider_job_id,
        )
