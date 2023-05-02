from asyncio import sleep
from typing import Dict, List, Optional
import requests

from edenai_apis.features import ProviderInterface, TextInterface, TranslationInterface
from edenai_apis.features.text import SummarizeDataClass, QuestionAnswerDataClass
from edenai_apis.features.translation import AutomaticTranslationDataClass
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.utils.conversion import concatenate_params_in_url
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType

class HuggingfaceApi(ProviderInterface, TextInterface, TranslationInterface):

    provider_name = "huggingface"
    base_url = "https://api-inference.huggingface.co/models"

    def __init__(self, api_keys: Dict = {}) -> None:
        self.api_key = load_provider(ProviderDataEnum.KEY, "huggingface", api_keys = api_keys)[
            "api_key"
        ]

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, url: str, inputs: dict):
        res = requests.post(url, headers=self.headers, json={"inputs": inputs})
        if res.status_code >= 500:
            raise ProviderException(message='Internal Server Error', code=res.status_code)
        return (res.status_code, res.json())

    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        """
        :param source_language:    String that contains language name of origin text
        :param target_language:    String that contains language name of origin text
        :param text:        String that contains input text to translate
        :return:            String that contains output result
        """

        model_types = [
            "Helsinki-NLP/opus-mt",
            "Helsinki-NLP/opus-mt-tc-big",
            "Helsinki-NLP/opus-tatoeba",
        ]
        for model_type in model_types:
            url = concatenate_params_in_url(
                url=f"{self.base_url}/{model_type}",
                params=[source_language, target_language],
                sep='-'
            )
            # url = format_string_url_language(
            #     f"{self.base_url}/{model_type}",
            #     source_language,
            #     "-",
            #     self.provider_name,
            #     False,
            # )
            # url = format_string_url_language(
            #     url, target_language, "-", self.provider_name, False
            # )

            status_code, response = self._post(url, text)

            # If the model does not handle the languages, check another type of models
            if "error" in response:
                if "does not exist" in response["error"]:
                    continue

            # Wait for model to load and _post again
            if "estimated_time" in response:
                print('sleeping for {response["estimated_time"]}')
                sleep(int(response["estimated_time"]) + 1)
                status_code, response = self._post(url, text)

            break

        if isinstance(response, dict) and response.get("error"):
            print(response["error"])
            raise ProviderException(response["error"])

        # Create output TextAutomaticTranslation object
        standardized: AutomaticTranslationDataClass

        # Getting translation
        result = response[0]
        if result["translation_text"] != "":
            standardized = AutomaticTranslationDataClass(text=result["translation_text"])

        return ResponseType[AutomaticTranslationDataClass](
            original_response=response,
            standardized_response=standardized
        )

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: str = None
    ) -> ResponseType[SummarizeDataClass]:

        """
        :param text:        String that contains input text
        :return:            String that contains output result
        """

        url = f"{self.base_url}/facebook/bart-large-cnn"

        status_code, response = self._post(url, text)

        standardized_response = SummarizeDataClass(
            result=response[0].get("summary_text")
        )

        return ResponseType[SummarizeDataClass](
            original_response=response,
            standardized_response=standardized_response
        )

    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str],
    ) -> ResponseType[QuestionAnswerDataClass]:
        """
        :param texts:       List of strings that contain that contains input text
        :question:          String that contains the question text
        :return:            String that contains output result
        """
        if model is None:
            model = "roberta-base-squad2"

        url = f"{self.base_url}/deepset/roberta-base-squad2"
        # Create standardized response
        status_code, response = self._post(url, {"question": question, "context": texts[0]})
        if status_code != 200:
            raise ProviderException(response.get('error'))

        data = response

        standardized_response = QuestionAnswerDataClass(answers=[data.get("answer")])

        return ResponseType[QuestionAnswerDataClass](
            original_response=response,
            standardized_response=standardized_response
        )
