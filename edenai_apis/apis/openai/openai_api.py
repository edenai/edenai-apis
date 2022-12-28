from typing import List, Optional, Sequence
import requests
import numpy as np

from edenai_apis.features import ProviderInterface, TextInterface
from edenai_apis.features.text import (
    SearchDataClass,
    InfosSearchDataClass,
    QuestionAnswerDataClass,
    SummarizeDataClass,
    AnonymizationDataClass,
    KeywordExtractionDataClass,
    SentimentAnalysisDataClass,
    InfosKeywordExtractionDataClass,
    TopicExtractionDataClass,
    ExtractedTopic,
    GenerationDataClass,
    CustomNamedEntityRecognitionDataClass,
    InfosCustomNamedEntityRecognitionDataClass,
)
from edenai_apis.features.translation import (
    LanguageDetectionDataClass,
    InfosLanguageDetectionDataClass,
    AutomaticTranslationDataClass,
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType


SCORE_MULTIPLIER = 100.0


class OpenaiApi(ProviderInterface, TextInterface):
    provider_name = "openai"

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.api_key = self.api_settings["api_key"]
        self.org_key = self.api_settings["org_key"]
        self.url = self.api_settings["url"]
        self.model = 'text-davinci-003'
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Organization": self.org_key,
            "Content-Type": "application/json",
        }
        self.max_tokens = 270

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

        standardized_response = SummarizeDataClass(
            result=original_response["choices"][0]["text"]
        )

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
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
        standardized_response = SearchDataClass(items=data_info_list)

        result = ResponseType[SearchDataClass](
            original_response=original_response,
            standardized_response=standardized_response
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
        context = document["standardized_response"]["items"][0]["document"]
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
        standardized_response = QuestionAnswerDataClass(answers=[answer])

        result = ResponseType[QuestionAnswerDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__anonymization(self, text: str, language: str) -> ResponseType[AnonymizationDataClass]:
        url = f"{self.url}/completions"
        prompt = f"Anoymize this text:\n\n"+text
        payload = {
        "prompt" : prompt,
        # "max_tokens" : self.max_tokens,
        "model" : self.model
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle povider error
        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])
        
        standardized_response = AnonymizationDataClass(result=original_response["choices"][0]['text'])

        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )

    def text__keyword_extraction(self, language: str, text: str) -> ResponseType[KeywordExtractionDataClass]:
        url = f"{self.url}/completions"
        prompt = f"Extract all keywords (keyword1,keyword2,keyword3) from this text: \n\n "+text+"\nkeywords:"
        payload = {
        "prompt" : prompt,
        "max_tokens" : self.max_tokens,
        "model" : self.model,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        # Handle povider error
        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])
        
        keywords = original_response['choices'][0]['text'].split(',')
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for keyword in keywords:
            items.append(
                InfosKeywordExtractionDataClass(keyword=keyword)
            )
        standardized_response = KeywordExtractionDataClass(items = items)

        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )
    
    def translation__language_detection(
        self, text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        url = f"{self.url}/completions"
        prompt = f"Detect the ISO 639-1 language (only the code) of this text: \n\n " + text + "\nISO 639-1:"
        payload = {
            "prompt" : prompt,
            "max_tokens" : self.max_tokens,
            "model" : self.model,
            "temperature" : 0,
            "logprobs":1,
        }

        original_response = requests.post(url, json=payload, headers=self.headers).json()
        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])
        
        items: Sequence[InfosLanguageDetectionDataClass] = []
        score = np.exp(original_response['choices'][0]['logprobs']['token_logprobs'][0])
        # replace are necessary to keep only language code
        isocode = original_response['choices'][0]['text'].replace(' ', '')
        items.append(
               InfosLanguageDetectionDataClass(
                    language=isocode,
                    display_name=get_language_name_from_code(isocode=isocode),
                   confidence = float(score)
               )
            )

        return ResponseType[LanguageDetectionDataClass](
            original_response=original_response,
            standardized_response=LanguageDetectionDataClass(items=items),
        )
        
    def translation__automatic_translation(
        self, source_language: str, target_language: str, text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        
        url = f"{self.url}/completions"
        prompt = f"Translate from {source_language} to {target_language} this text:\n\n{text}\n\ntranslation:"
        payload = {
        "prompt" : prompt,
        "max_tokens" : self.max_tokens,
        "model" : self.model,
        "temperature" : 0,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])

        standardized = AutomaticTranslationDataClass(text=original_response['choices'][0]['text'])

        return ResponseType[AutomaticTranslationDataClass](
            original_response=original_response, standardized_response=standardized.dict()
        )
    
    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        url = f"{self.url}/completions"
        prompt = f"Label the text with one of these sentiments 'Positive','Negative','Neutral':\n\n text:"+text+"\nlabel:"
        payload = {
        "prompt" : prompt,
        # "max_tokens" : self.max_tokens,
        "model" : self.model,
        "temperature" : 0,
        "logprobs":1,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()

        # Create output response
        # Get score 
        score = np.exp(original_response['choices'][0]['logprobs']['token_logprobs'][0])
        standarize = SentimentAnalysisDataClass(
            general_sentiment=original_response['choices'][0]['text'][1:],
            general_sentiment_rate = float(score)
            )

        return ResponseType[SentimentAnalysisDataClass](
            original_response=original_response, standardized_response=standarize
        )
        
    def text__topic_extraction(
        self, language: str, text: str
    ) -> ResponseType[TopicExtractionDataClass]:
        url = f"{self.url}/completions"

        prompt = f"What is the main taxonomy of the text:"+text+"please put the result in this line:\n\n"
        payload = {
        "prompt" : prompt,
        "max_tokens" : self.max_tokens,
        "model" : self.model,
        "logprobs":1,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        # Create output response
        # Get score 
        score = np.exp(original_response['choices'][0]['logprobs']['token_logprobs'][0])
        categories: Sequence[ExtractedTopic] = []
        categories.append(ExtractedTopic(
            category = original_response['choices'][0]['text'], importance = float(score)
        ))
        standarized_response = TopicExtractionDataClass(
            items = categories
            )

        return ResponseType[TopicExtractionDataClass](
            original_response=original_response, standardized_response=standarized_response
        )
        
    def text__generation(
        self, text : str, 
        temperature : float, 
        max_tokens : int,
        model : Optional[str] = None,
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.url}/completions"
        
        if not model :
            model = 'text-davinci-003'
            
        payload = {
            "prompt": text,
            "model" : model,
            "temperature" : temperature,
        }
        if max_tokens !=0:
            payload['max_tokens'] = max_tokens
            
        original_response = requests.post(url, json=payload, headers= self.headers).json()
        
        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])
        
        standardized_response = GenerationDataClass(
            generated_text = original_response['choices'][0]['text']
        )
        return ResponseType[GenerationDataClass](
            original_response=original_response,
            standardized_response = standardized_response
        )
        
    def text__custom_named_entity_recognition(self, text: str, entities: List[str]
                                              ) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        url = f"{self.url}/completions"
        built_entities = ','.join(entities)
        prompt = f"Extract these entities ({built_entities}) from this text and format the result as entity:result separated by ;\ntext:"+text
        payload = {
        "prompt" : prompt,
        "model" : self.model,
        "temperature" : 0.0,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        # Handle povider error
        if "error" in original_response:
            raise ProviderException(original_response["error"]["message"])
        
        items: Sequence[InfosCustomNamedEntityRecognitionDataClass] = []
        entities = original_response['choices'][0]['text'].replace("\n", "").split(';')
        for entity in entities:
            item = entity.split(':')
            items.append(InfosCustomNamedEntityRecognitionDataClass(
                entity = item[1],
                category = item[0]
            ))
            

        standardized_response = CustomNamedEntityRecognitionDataClass(items=items)

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )