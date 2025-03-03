import base64
import json
import mimetypes
import os
import re
import uuid
from io import BytesIO
from typing import Dict, List, Literal, Optional, Type, Union

import httpx
from loaders.data_loader import ProviderDataEnum
from loaders.loaders import load_provider
from pydantic import BaseModel

from edenai_apis.features.image import (
    ExplicitContentDataClass,
    GeneratedImageDataClass,
    GenerationDataClass,
    LogoDetectionDataClass,
    QuestionAnswerDataClass,
)
from edenai_apis.features.multimodal.chat import (
    ChatDataClass as ChatMultimodalDataClass,
)
from edenai_apis.features.multimodal.chat import (
    ChatStreamResponse as ChatMultimodalStreamResponse,
)
from edenai_apis.features.multimodal.chat import StreamChat as StreamMultimodalChat
from edenai_apis.features.text import (
    AnonymizationDataClass,
    CodeGenerationDataClass,
    CustomClassificationDataClass,
    CustomNamedEntityRecognitionDataClass,
    EmbeddingDataClass,
    EmbeddingsDataClass,
    KeywordExtractionDataClass,
    ModerationDataClass,
    NamedEntityRecognitionDataClass,
    SentimentAnalysisDataClass,
    SpellCheckDataClass,
    SummarizeDataClass,
    TextModerationItem,
    TopicExtractionDataClass,
)
from edenai_apis.features.text.chat import ChatDataClass, ChatMessageDataClass
from edenai_apis.features.text.chat.chat_dataclass import (
    ChatStreamResponse,
    StreamChat,
    ToolCall,
)
from edenai_apis.features.text.moderation.category import (
    CategoryType as CategoryTypeModeration,
)
from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    LanguageDetectionDataClass,
)
from edenai_apis.llmengine.clients import LLM_COMPLETION_CLIENTS
from edenai_apis.llmengine.clients.completion import CompletionClient
from edenai_apis.llmengine.mapping import Mappings
from edenai_apis.llmengine.prompts import BasePrompt
from edenai_apis.llmengine.types.response_types import ResponseModel
from edenai_apis.llmengine.utils.moderation import moderate
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.upload_s3 import upload_file_bytes_to_s3


class LLMEngine:
    """
    Instantiate the engine for doing calls to LLMs
    """

    def __init__(
        self,
        provider_name: str,
        model: Optional[str] = None,
        client_name: Optional[str] = None,
        application_name: str = uuid.uuid4(),
        provider_config: dict = {},
        **kwargs,
    ) -> None:
        # Set the user
        self.model = model
        self.provider_name = provider_name
        self.application_name = str(application_name)
        if client_name is None:
            client_name = next(iter(LLM_COMPLETION_CLIENTS))
        # TODO change the completion client to behave in the same way
        self.provider_config = provider_config
        self.completion_client: CompletionClient = LLM_COMPLETION_CLIENTS[client_name](
            model_name=model, provider_name=self.provider_name
        )

    def _prepare_args(self, model: str, **kwargs) -> Dict:
        params = {
            "model": model,
            "drop_params": True,
        }
        params.update(self.provider_config)
        params.update(kwargs)
        return params

    def _execute_completion(self, params: Dict, response_class: Type, **kwargs):
        try:
            params.pop("moderate_content", None)
            response = self.completion_client.completion(**params, **kwargs)
            response = ResponseModel.model_validate(response)
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc
        standardized_response = response_class(**result)
        return ResponseType[response_class](
            original_response=response.to_dict(),
            standardized_response=standardized_response,
            usage=response.usage,
        )

    @moderate
    def chat(
        self,
        text: str,
        chatbot_global_action: Optional[str],
        previous_history: Optional[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        model: str,
        stream=False,
        available_tools: Optional[List[dict]] = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[ChatDataClass, StreamChat]]:
        previous_history = previous_history or []
        messages = Mappings.format_chat_messages(
            text, chatbot_global_action, previous_history, tool_results
        )
        call_params = self._prepare_args(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if available_tools and not tool_results:
            call_params["tools"] = Mappings.convert_tools_to_openai(
                tools=available_tools
            )
            call_params["tool_choice"] = tool_choice
        response = self.completion_client.completion(**call_params, **kwargs)
        if stream is False:
            response = ResponseModel.model_validate(response)
            message = response.choices[0].message
            generated_text = message.content
            original_tool_calls = message.tool_calls or []
            tool_calls = []
            for call in original_tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=call["id"],
                        name=call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                )
            messages = [
                ChatMessageDataClass(role="user", message=text, tools=available_tools),
                ChatMessageDataClass(
                    role="assistant",
                    message=generated_text,
                    tool_calls=tool_calls,
                ),
            ]
            messages_json = [m.dict() for m in messages]

            standardized_response = ChatDataClass(
                generated_text=generated_text, message=messages_json
            )

            return ResponseType[ChatDataClass](
                original_response=response.to_dict(),
                standardized_response=standardized_response,
                usage=response.usage,
            )
        else:
            stream_response = (
                ChatStreamResponse(
                    text=chunk.choices[0].delta.content or "",
                    blocked=False,
                    provider=self.provider_name,
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamChat](
                original_response=None,
                standardized_response=StreamChat(stream=stream_response),
            )

    @moderate
    def multimodal_chat(
        self,
        messages: List[ChatMessageDataClass],
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
        tool_choice: Literal["auto", "required", "none"] = "auto",
        available_tools: Optional[dict] = None,
        tool_results: Optional[List[dict]] = None,
        **kwargs,
    ) -> ResponseType[Union[ChatMultimodalDataClass, StreamMultimodalChat]]:

        transformed_messages = Mappings.format_multimodal_messages(
            messages=messages, system_prompt=chatbot_global_action
        )
        args = self._prepare_args(
            model=model,
            messages=transformed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            top_p=top_p,
            stream=stream,
        )

        if available_tools and len(available_tools) > 0 and not tool_results:
            args["tools"] = Mappings.convert_tools_llmengine(available_tools)
            args["tool_choice"] = tool_choice

        args["response_format"] = response_format
        args["drop_invalid_params"] = True
        response = self.completion_client.completion(**args, **kwargs)
        if stream is False:
            response = ResponseModel.model_validate(response)
            generated_text = (
                response.choices[0].message.content or "" if response.choices else ""
            )

            standardized_response = (
                ChatMultimodalDataClass.generate_standardized_response(
                    generated_text=generated_text, messages=messages
                )
            )
            return ResponseType[ChatMultimodalDataClass](
                original_response=response.to_dict(),
                standardized_response=standardized_response,
                usage=response.usage,
            )

        else:
            stream_response = (
                ChatMultimodalStreamResponse(
                    text=chunk["choices"][0]["delta"].get("content") or "",
                    blocked=not chunk["choices"][0].get("finish_reason")
                    in (None, "stop"),
                    provider=self.provider_name,
                )
                for chunk in response
                if chunk
            )

            return ResponseType[StreamMultimodalChat](
                original_response=None,
                standardized_response=StreamMultimodalChat(stream=stream_response),
            )

    def summarize(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[SummarizeDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="",
            example_file="text/summarize/summarize_response.json",
            dataclass=SummarizeDataClass,
        )
        messages.append({"role": "user", "content": text + "\nTLDR:"})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(params=args, response_class=SummarizeDataClass)

    def logo_detection(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ) -> ResponseType[LogoDetectionDataClass]:
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as fstream:
            base64_data = base64.b64encode(fstream.read()).decode("utf-8")
        messages = BasePrompt.compose_prompt(
            behavior="You are a Logo Detection model. You get an image input and return logos detected inside it. If no logo is detected the items list should be empty",
            example_file="image/logo_detection/logo_detection_response.json",
            dataclass=LogoDetectionDataClass,
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                    },
                ],
            }
        )
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=LogoDetectionDataClass
        )

    @moderate
    def image_qa(
        self,
        image_data: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
        question: Optional[str] = None,
        **kwargs,
    ) -> ResponseType[QuestionAnswerDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a visual question-answering assistant that analyzes images and responds to questions about them.",
            example_file="image/question_answer/question_answer_response.json",
            dataclass=LogoDetectionDataClass,
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question or "Describe the following image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data},
                    },
                ],
            }
        )
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=QuestionAnswerDataClass
        )

    def image_moderation(
        self, file: str, file_url: str = "", model: Optional[str] = None, **kwargs
    ):
        mime_type = mimetypes.guess_type(file)[0]
        with open(file, "rb") as fstream:
            base64_data = base64.b64encode(fstream.read()).decode("utf-8")
        messages = BasePrompt.compose_prompt(
            behavior="You are an Explicit Image Detection model.",
            example_file="image/explicit_content/explicit_content_response.json",
            dataclass=ExplicitContentDataClass,
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                    },
                ],
            }
        )
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=ExplicitContentDataClass
        )

    def sentiment_analysis(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[SentimentAnalysisDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a Text Sentiment Analysis model. You extract the sentiment of a textual input.",
            example_file="text/sentiment_analysis/sentiment_analysis_response.json",
            dataclass=SentimentAnalysisDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=SentimentAnalysisDataClass
        )

    def keyword_extraction(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[KeywordExtractionDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are an Keyword Extract model. You extract keywords from a text input.",
            example_file="text/keyword_extraction/keyword_extraction_response.json",
            dataclass=KeywordExtractionDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=KeywordExtractionDataClass
        )

    def spell_check(self, text: str, model: str, **kwargs):
        messages = BasePrompt.compose_prompt(
            behavior="You are a Spell Checking model. You analyze a text input and proficiently detect and correct any grammar, syntax, spelling or other types of errors.",
            example_file="text/spell_check/spell_check_response.json",
            dataclass=SpellCheckDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(params=args, response_class=SpellCheckDataClass)

    def topic_extraction(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[TopicExtractionDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a Text Topic Exstraction Model. You extract the main topic of a textual input.",
            example_file="text/topic_extraction/topic_extraction_response.json",
            dataclass=TopicExtractionDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=TopicExtractionDataClass
        )

    def named_entity_recognition(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[NamedEntityRecognitionDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a Named Entity Extraction Model. Given an input text you should extract extract all entities in it.",
            example_file="text/named_entity_recognition/named_entity_recognition_response.json",
            dataclass=NamedEntityRecognitionDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=NamedEntityRecognitionDataClass
        )

    def pii(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[AnonymizationDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a PII system that takes a text input containing personally identifiable information (PII) and generates an anonymized version of the text. The length of each entity must be exactly equals to the number of characters of the identified entity not including punctuation marks",
            example_file="text/anonymization/anonymization_response.json",
            dataclass=AnonymizationDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=AnonymizationDataClass
        )

    def code_generation(
        self,
        instruction: str,
        temperature: float,
        max_tokens: int,
        model: str,
        **kwargs,
    ) -> ResponseType[CodeGenerationDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a Code Generation Model. Given a text input, you should generate a code output.",
            example_file="text/code_generation/code_generation_response.json",
            dataclass=CodeGenerationDataClass,
        )
        messages.append({"role": "user", "content": instruction})
        args = self._prepare_args(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=CodeGenerationDataClass
        )

    def embeddings(
        self, texts: List[str], model: str, **kwargs
    ) -> ResponseType[EmbeddingsDataClass]:
        args = {
            "model": model,
            "input": texts,
            "drop_params": True,
        }
        args.update(self.provider_config)
        response = self.completion_client.embedding(**args, **kwargs)
        # response = EmbeddingResponseModel.model_validate(response)
        items = []
        embeddings = response.get("data", [{}])
        for embedding in embeddings:
            items.append(EmbeddingDataClass(embedding=embedding["embedding"]))
        standardized_response = EmbeddingsDataClass(items=items)
        return ResponseType[EmbeddingsDataClass](
            original_response=response.to_dict(),
            standardized_response=standardized_response,
            usage=response.usage,
        )

    def moderation(self, text: str, **kwargs) -> ResponseType[ModerationDataClass]:
        # Only availaible for OpenAI
        args = {"input": text}
        args.update(self.provider_config)
        response = self.completion_client.moderation(**args, **kwargs)
        classification = []
        result = response.results
        if result:
            category_scores = result[0].category_scores
            for key, value in category_scores.to_dict().items():
                classificator = CategoryTypeModeration.choose_category_subcategory(key)
                classification.append(
                    TextModerationItem(
                        label=key,
                        category=classificator["category"],
                        subcategory=classificator["subcategory"],
                        likelihood_score=value,
                        likelihood=standardized_confidence_score(value),
                    )
                )
        standardized_response: ModerationDataClass = ModerationDataClass(
            nsfw_likelihood=ModerationDataClass.calculate_nsfw_likelihood(
                classification
            ),
            items=classification,
            nsfw_likelihood_score=ModerationDataClass.calculate_nsfw_likelihood_score(
                classification
            ),
        )
        return ResponseType[ModerationDataClass](
            original_response=response.to_dict(),
            standardized_response=standardized_response,
        )

    def custom_classification(
        self,
        texts: List[str],
        labels: List[str],
        model: str,
        examples: List[List[str]],
        **kwargs,
    ) -> ResponseType[CustomClassificationDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior=" You are a Text Classification Model. Given a list of possible labels and a list of texts, you should classify each text by giving it one label.",
            example_file="text/custom_classification/custom_classification_response.json",
            dataclass=CustomClassificationDataClass,
        )
        formated_examples = Mappings.format_classification_examples(examples)

        text = f"""
                Possible Labels : 
                {labels}

                {formated_examples}
                ======
                List of texts : 

                {texts}
                """.format(
            labels, formated_examples, texts
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=CustomClassificationDataClass
        )

    def custom_named_entity_recognition(
        self,
        text: str,
        entities: List[str],
        model: str,
        examples: Optional[List[Dict]] = None,
        **kwargs,
    ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        example_section = Mappings.format_ner_examples(examples=examples)
        formatted_entities = ", ".join(entities)
        instruction_message = f"list of entities types to extract : {formatted_entities}.\n {example_section  if example_section else ''}"
        messages = BasePrompt.compose_prompt(
            behavior="You are a Named Entity Extraction Model. Given a list of Entities Types and a text input, you should extract extract all entities of the given entities types."
            + instruction_message,
            example_file="text/custom_classification/custom_classification_response.json",
            dataclass=CustomNamedEntityRecognitionDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=CustomNamedEntityRecognitionDataClass
        )

    def language_detection(
        self, text: str, model: str, **kwargs
    ) -> ResponseType[LanguageDetectionDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior="You are a language detection model capable of automatically identifying the language of a given text",
            example_file="translation/language_detection/language_detection_response.json",
            dataclass=LanguageDetectionDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=LanguageDetectionDataClass
        )

    def automatic_translation(
        self,
        source_language: str,
        target_language: str,
        text: str,
        model: str,
        **kwargs,
    ) -> ResponseType[AutomaticTranslationDataClass]:
        messages = BasePrompt.compose_prompt(
            behavior=f"You are a translation model capable of translating text from {source_language} to {target_language} with high accuracy and fluency.",
            example_file="translation/automatic_translation/automatic_translation_response.json",
            dataclass=AutomaticTranslationDataClass,
        )
        messages.append({"role": "user", "content": text})
        args = self._prepare_args(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return self._execute_completion(
            params=args, response_class=AutomaticTranslationDataClass
        )

    @moderate
    def image_generation(
        self, prompt: str, model: str, resolution: str, n: int, **kwargs
    ):
        args = {
            "model": model,
            "prompt": prompt,
            "size": resolution,
            "n": n,
            "response_format": "b64_json",
        }
        args.update(self.provider_config)
        response = self.completion_client.image_generation(**args, **kwargs)
        generations = []
        for generated_image in response.data:
            image_b64 = generated_image.b64_json
            image_data = image_b64.encode()
            image_content = BytesIO(base64.b64decode(image_data))
            resource_url = upload_file_bytes_to_s3(
                image_content, ".png", "users_process"
            )
            generations.append(
                GeneratedImageDataClass(
                    image=image_b64, image_resource_url=resource_url
                )
            )
        standardized_response = GenerationDataClass(items=generations)
        return ResponseType[GenerationDataClass](
            original_response=response.to_dict(),
            standardized_response=standardized_response,
            usage=response.usage,
        )


class StdLLMEngine(LLMEngine):

    PROVIDER_MAPPING = {
        "vertex_ai": "google",
        "gemini": "google",
        "bedrock_converse": "bedrock",
        "amazon": "bedrock",
    }

    def __init__(
        self,
        provider_config: dict = {},
        **kwargs,
    ):
        super().__init__(
            model=None,
            client_name="litellm",
            application_name="std_chat",
            provider_config=provider_config,
            provider_name=None,
            **kwargs,
        )

    @staticmethod
    def map_provider(provider_name: str) -> str:
        if provider_name is None:
            return None
        # Try to regex match the keys of PROVIDER_MAPPING and provider_name. The first one ot match wins
        for key in StdLLMEngine.PROVIDER_MAPPING.keys():
            if re.match(key, provider_name, re.RegexFlag.IGNORECASE):
                return StdLLMEngine.PROVIDER_MAPPING[key]
        return provider_name

    def completion(
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
    ):
        if "provider" in kwargs:
            # Verify if the provider is gemini
            provider_name = kwargs.pop("provider", None)
            is_gemini = provider_name == "gemini"
            provider_name = StdLLMEngine.map_provider(provider_name)
            if provider_name == "google" and not is_gemini:
                api_settings, location = load_provider(
                    ProviderDataEnum.KEY,
                    provider_name=provider_name,
                    location=True,
                    api_keys=api_key,
                )
                self.project_id = api_settings["project_id"]
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = location
            elif is_gemini:
                api_settings = load_provider(
                    ProviderDataEnum.KEY, provider_name=provider_name, api_keys=api_key
                )
                api_key = api_settings["genai_api_key"]
            else:
                api_settings = load_provider(
                    ProviderDataEnum.KEY, provider_name=provider_name, api_keys=api_key
                )
                api_key = api_settings["api_key"]
        try:
            completion_params = {
                "messages": messages,
                "model": model,
                "timeout": timeout,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stream": stream,
                "stream_options": stream_options,
                "stop": stop,
                "stop_sequences": stop_sequences,
                "max_tokens": max_tokens,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "response_format": response_format,
                "seed": seed,
                "tools": tools,
                "tool_choice": tool_choice,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "parallel_tool_calls": parallel_tool_calls,
                "deployment_id": deployment_id,
                "extra_headers": extra_headers,
                "functions": functions,
                "function_call": function_call,
                "base_url": base_url,
                "api_version": api_version,
                "api_key": api_key,
                "model_list": model_list,
                "drop_invalid_params": drop_invalid_params,
                "user": user,
                **kwargs,
            }
            response = self._execute_completion(completion_params)
            return response
        except Exception as ex:
            raise ex

    def _execute_completion(self, params: Dict, **kwargs):
        try:
            response = self.completion_client.completion(**params, **kwargs)
            response = ResponseModel.model_validate(response)
            return response
        except Exception as ex:
            raise ex
