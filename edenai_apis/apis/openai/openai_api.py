from typing import List, Optional, Sequence
import requests
import numpy as np
import json
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import get_language_name_from_code
from edenai_apis.utils.types import ResponseType
from edenai_apis.features import ProviderInterface, TextInterface, ImageInterface
from edenai_apis.features.text.question_answer import QuestionAnswerDataClass
from edenai_apis.features.text.summarize import SummarizeDataClass
from edenai_apis.features.text.anonymization import AnonymizationDataClass
from edenai_apis.features.text.sentiment_analysis import SentimentAnalysisDataClass
from edenai_apis.features.text.generation import GenerationDataClass
from edenai_apis.features.text.search import SearchDataClass, InfosSearchDataClass
from edenai_apis.features.text.keyword_extraction import KeywordExtractionDataClass,InfosKeywordExtractionDataClass
from edenai_apis.features.text.topic_extraction import TopicExtractionDataClass, ExtractedTopic
from edenai_apis.features.text.custom_named_entity_recognition import  CustomNamedEntityRecognitionDataClass, InfosCustomNamedEntityRecognitionDataClass
from edenai_apis.features.text.custom_classification import CustomClassificationDataClass, ItemCustomClassificationDataClass
from edenai_apis.features.text.moderation import   ModerationDataClass, TextModerationItem
from edenai_apis.features.translation.automatic_translation import AutomaticTranslationDataClass
from edenai_apis.features.translation.language_detection import LanguageDetectionDataClass, InfosLanguageDetectionDataClass
from edenai_apis.features.image.generation import (
    GenerationDataClass as ImageGenerationDataClass,
    GeneratedImageDataClass
)
from .helpers import (
    construct_search_context,
    get_score,
    construct_anonymization_context,
    construct_classification_instruction,
    format_example_fn,
    check_openai_errors,
    construct_keyword_extraction_context,
    construct_language_detection_context,
    construct_translation_context,
    construct_sentiment_analysis_context,
    construct_topic_extraction_context,
)

SCORE_MULTIPLIER = 100.0


class OpenaiApi(ProviderInterface, TextInterface, ImageInterface):
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

    
    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: Optional[str]
    ) -> ResponseType[SummarizeDataClass]:

        if not model:
            model = self.model

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

        # Handle errors
        check_openai_errors(original_response)

        standardized_response = SummarizeDataClass(
            result=original_response["choices"][0]["text"]
        )

        result = ResponseType[SummarizeDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result

    def text__moderation(self, text: str, language: str
    ) -> ResponseType[ModerationDataClass]:
        try:
            response = requests.post(
                f"{self.url}/moderations",
                headers= self.headers,
                json={
                    "input" : text
                }
            )
        except Exception as exc:
            raise ProviderException(str(exc))
        original_response = response.json()
        
        # Handle errors
        check_openai_errors(original_response)
        
        classification : Sequence[TextModerationItem] = []
        if result := original_response.get("results", None):
            for key, value in result[0].get("category_scores", {}).items():
                classification.append(
                    TextModerationItem(
                        label= key,
                        likelihood= standardized_confidence_score(value)
                    )
                )
        standardized_response : ModerationDataClass = ModerationDataClass(
            nsfw_likelihood= ModerationDataClass.calculate_nsfw_likelihood(classification),
            items= classification
        )

        return ResponseType[ModerationDataClass](
            original_response= original_response,
            standardized_response= standardized_response
        )
        
    def text__search(
        self, texts: List[str], query: str, model: str = None
    ) -> ResponseType[SearchDataClass]:
        if model is None:
            model = "text-davinci-003"

        prompts = [construct_search_context(query, doc) for doc in [""] + texts]

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

        # Handle errors
        check_openai_errors(original_response)

        resps_by_index = {
            choice["index"]: choice for choice in original_response["choices"]
        }
        scores = [
            get_score(
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

        # Handle errors
        check_openai_errors(original_response)
        
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
        prompt = construct_anonymization_context(text)
        payload = {
            "prompt": prompt,
            "model" : self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)
        
        data = original_response['choices'][0]['text'].replace("\n\n", " ").strip()
        data_dict = json.loads(fr"{data}")
        standardized_response = AnonymizationDataClass(result=data_dict.get('redactedText'))
        return ResponseType[AnonymizationDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )

    def text__keyword_extraction(self, language: str, text: str) -> ResponseType[KeywordExtractionDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_keyword_extraction_context(text)
        payload = {
        "prompt" : prompt,
        "max_tokens" : self.max_tokens,
        "model" : self.model,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)
        
        data = original_response['choices'][0]['text'].replace("\n", " ").strip()
        data_list = json.loads(fr"{data}")
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for keyword in data_list:
           items.append(
               InfosKeywordExtractionDataClass(
                   keyword=keyword.get('keyword'),
                   importance=keyword.get('score')),
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
        prompt = construct_language_detection_context(text)
        payload = {
            "prompt" : prompt,
            "max_tokens" : self.max_tokens,
            "model" : self.model,
            "temperature" : 0,
            "logprobs":1,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)
        
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
        prompt=construct_translation_context(text, source_language, target_language)
        payload = {
        "prompt" : prompt,
        "max_tokens" : self.max_tokens,
        "model" : self.model,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)

        standardized = AutomaticTranslationDataClass(text=original_response['choices'][0]['text'])

        return ResponseType[AutomaticTranslationDataClass](
            original_response=original_response, standardized_response=standardized.dict()
        )
    
    def text__sentiment_analysis(
        self, language: str, text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        url = f"{self.url}/completions"
        prompt = construct_sentiment_analysis_context(text)
        payload = {
        "prompt" : prompt,
        "model" : self.model,
        "temperature" : 0,
        "logprobs":1,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)
        
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

        prompt = construct_topic_extraction_context(text)
        payload = {
        "prompt" : prompt,
        "max_tokens" : self.max_tokens,
        "model" : self.model,
        "logprobs":1,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()

        # Handle errors
        check_openai_errors(original_response)
        
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
        
        # Handle errors
        check_openai_errors(original_response)
        
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
        Examples = f"Text : Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company. Originally marketed as a temperance drink and intended as a patent medicine, it was invented in the late 19th century by John Stith Pemberton in Atlanta, Georgia.\n\nEntities : person, state, drink, date.\n\nExtracted_Entities : person : John Stith Pemberton; state : Georgia; drink : Coca-Cola; drink : coke; date : 19th century."
        prompt = f"{Examples}\n\nText : "+text+" \n\nEntities :"+built_entities+"\n\nExtracted_Entities :"
        payload = {
        "prompt" : prompt,
        "model" : self.model,
        "temperature" : 0.0,
        "top_p": 1,
        "max_tokens":250,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)
        
        items: Sequence[InfosCustomNamedEntityRecognitionDataClass] = []
        entities = original_response['choices'][0]['text'].replace("\n", "").split(';')
        for entity in entities:
            item = entity.split(':')
            detected_entities = item[1].split(',')
            if len(detected_entities)>1:
                for ent in detected_entities:
                    items.append(InfosCustomNamedEntityRecognitionDataClass(
                    entity = item[0],
                    category = ent
                ))
            else:
                items.append(InfosCustomNamedEntityRecognitionDataClass(
                entity = item[0],
                category = item[1]))

        standardized_response = CustomNamedEntityRecognitionDataClass(items=items)

        return ResponseType[CustomNamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )

    def text__custom_classification(
        self,
        texts: List[str],
        labels: List[str],
        examples: List[List[str]]
    ) -> ResponseType[CustomClassificationDataClass]:
        url = f"{self.url}/completions"

        # Construct prompt 
        instruction = construct_classification_instruction(labels)
        example_prompts = format_example_fn(examples) 
        inputs_prompts = [
            f"{input}\n" for input in texts
        ]
        prompt = example_prompts + instruction + "".join(inputs_prompts)+ "\n\nText categories:\n\n"

        # Build the request
        payload = {
        "prompt" : prompt,
        "model" : self.model,
        "top_p": 1,
        "max_tokens":250,
        "logprobs": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()

        # Handle errors
        check_openai_errors(original_response)

        # Getting labels 
        detected_labels = original_response['choices'][0]['text'].split('\n')
        # Calculate scores
        scores = []
        score = 0
        logprobs = original_response['choices'][0]['logprobs']['top_logprobs']
        for prob in logprobs:
            score = score + list(prob.values())[0]
            if prob.get("\n") or prob.get("<|endoftext|>"):
                scores.append(np.exp(score))
                score = 0

        classifications = []
        for (input, label, score) in zip(texts, detected_labels, scores):
            classifications.append(
                ItemCustomClassificationDataClass(
                    input = input,
                    label = label,
                    confidence = float(score),
                )
            )

        return ResponseType[CustomClassificationDataClass](
            original_response=original_response,
            standardized_response=CustomClassificationDataClass(classifications=classifications))
        
    def image__generation(
        self, text: str, num_images : int = 1, width : int = 512, height : int = 512
    ) -> ResponseType[ImageGenerationDataClass]:
        url = f"{self.url}/images/generations"
        size = f"{width}x{height}"
        payload = {
            "prompt" : text,
            "n" : num_images,
            "size" : size,
            "response_format" : "b64_json"
        }
        original_response = requests.post(url, json=payload, headers=self.headers).json()
        
        # Handle errors
        check_openai_errors(original_response)
        
        generations: Sequence[GeneratedImageDataClass] = []
        for generated_image in original_response.get('data'):
            generations.append(GeneratedImageDataClass(
                image=generated_image.get('b64_json')))
            
        return ResponseType[ImageGenerationDataClass](
            original_response=original_response,
            standardized_response=ImageGenerationDataClass(items=generations))