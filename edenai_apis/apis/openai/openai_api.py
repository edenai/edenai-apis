from typing import List, Optional
import requests

from edenai_apis.features import ProviderApi, Text
from edenai_apis.features.text import (
    SearchDataClass,
    InfosSearchDataClass,
    QuestionAnswerDataClass,
    SummarizeDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


SCORE_MULTIPLIER = 100.0


class OpenaiApi(ProviderApi, Text):
    provider_name = "openai"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.org_key = self.api_settings["org_key"]
        self.url = self.api_settings["url"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_key,
            "Content-Type": "application/json",
        }
        self.max_tokens = 250

    @staticmethod
    def _construct_context(query, document) -> str:
        return f"<|endoftext|>{document}\n\n---\n\nThe above passage is related to: {query}"

    @staticmethod
    def _get_score(context, query, log_probs, text_offsets) -> float:
        log_prob = 0
        count = 0
        cutoff = len(context) - len(query)

        for i in range(len(text_offsets) - 1, 0, -1):
            log_prob += log_probs[i]
            count += 1

            if text_offsets[i] <= cutoff and text_offsets[i] != text_offsets[i - 1]:
                break

        return log_prob / float(count) * SCORE_MULTIPLIER

    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: Optional[str]
    ) -> ResponseType[SummarizeDataClass]:

        if not model:
            model = "text-davinci-003"

        url = f"{self.url}/engines/{model}/completions"
        payload = {
            "prompt": text + "\n\nTl;dr",
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])

        standarized_response = SummarizeDataClass(
            result=original_response["choices"][0]["text"]
        )

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result

    def text__search(
        self, texts: List[str], query: str, model: str = None
    ) -> ResponseType[SearchDataClass]:
        if model is None:
            model = "text-davinci-003"

        prompts = [OpenaiApi._construct_context(query, doc) for doc in [""] + texts]

        url = f"{self.url}/completions"
        payload = {
            "model": model,
            "prompt": prompts,
            "max_tokens": 6,
            "temperature": 0,
            "top_p": 1,
            "logprobs": 0,
            "n": 1,
            "stop": "\n",
            "echo": True,
        }

        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])

        resps_by_index = {
            choice["index"]: choice for choice in original_response["choices"]
        }
        scores = [
            OpenaiApi._get_score(
                prompts[i],
                query,
                resps_by_index[i]["logprobs"]["token_logprobs"],
                resps_by_index[i]["logprobs"]["text_offset"],
            )
            for i in range(len(prompts))
        ]
        # Process results
        scores = [score - scores[0] for score in scores][1:]

        data_info_list = []
        for document_idx, score in enumerate(scores):
            data_info_list.append(
                InfosSearchDataClass(
                    object="search_result",
                    document=document_idx,
                    score=round(score, 3),
                )
            )
        standarized_response = SearchDataClass(items=data_info_list)

        result = ResponseType[SearchDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )
        return result

    def text__question_answer(
        self,
        texts: List[str],
        question: str,
        temperature: float,
        examples_context: str,
        examples: List[List[str]],
        model: Optional[str],
    ) -> ResponseType[QuestionAnswerDataClass]:
        if not model:
            model = "text-davinci-003"

        url = f"{self.url}/completions"

        # With search get the top document with the question & construct the context
        document = self.text__search(texts, question).dict()
        context = document["standarized_response"]["items"][0]["document"]
        prompt_questions = [
            "\nQ:" + example[0] + "\nA:" + example[1] for example in examples
        ]
        prompts = [
            examples_context
            + "\n\n"
            + "".join(prompt_questions)
            + "\n\n"
            + texts[context]
            + "\nQ:"
            + question
            + "\nA:"
        ]
        payload = {
            "model": model,
            "prompt": prompts,
            "max_tokens": 100,
            "temperature": temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        original_response = requests.post(
            url, json=payload, headers=self.headers
        ).json()

        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])
        answer = original_response["choices"][0]["text"].split("\n")
        answer = answer[0]
        standarized_response = QuestionAnswerDataClass(answers=[answer])

        result = ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standarized_response=standarized_response,
        )
        return result
