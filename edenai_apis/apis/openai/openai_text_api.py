from typing import List, Optional, Sequence, Dict
import requests
import numpy as np
import json
from edenai_apis.features.text.code_generation.code_generation_dataclass import CodeGenerationDataClass
from edenai_apis.features.text.named_entity_recognition.named_entity_recognition_dataclass import NamedEntityRecognitionDataClass
from edenai_apis.features.text.spell_check.spell_check_dataclass import SpellCheckDataClass, SpellCheckItem
from edenai_apis.utils.conversion import standardized_confidence_score
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType
from edenai_apis.features import TextInterface
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
from edenai_apis.features.text.embeddings import EmbeddingDataClass, EmbeddingsDataClass
from edenai_apis.features.text.chat import ChatDataClass, ChatMessageDataClass
from .helpers import (
    construct_ner_instruction,
    construct_search_context,
    construct_spell_check_instruction,
    get_score,
    construct_anonymization_context,
    construct_classification_instruction,
    format_example_fn,
    check_openai_errors,
    construct_keyword_extraction_context,
    construct_sentiment_analysis_context,
    construct_topic_extraction_context,
    construct_custom_ner_instruction
)
class OpenaiTextApi(TextInterface):
    def text__summarize(
        self, text: str, output_sentences: int, language: str, model: str
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
        try:
            original_response = requests.post(url, json=payload, headers=self.headers).json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error") from exc
        
        # Handle errors
        check_openai_errors(original_response)
        
        try:
            keywords = json.loads(original_response['choices'][0]['text'])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException("An error occurred while parsing the response.") from exc
        
        standardized_response = KeywordExtractionDataClass(items = keywords['items'])
        return ResponseType[KeywordExtractionDataClass](
            original_response=original_response,
            standardized_response=standardized_response
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
    

    def text__code_generation(
        self,
        instruction: str,
        temperature: float,
        max_tokens: int,
        prompt: str = ""
    ) -> ResponseType[CodeGenerationDataClass]:
         
        url = f"{self.url}/chat/completions"
        model = "gpt-3.5-turbo"

        messages = [
                {"role": "system", "content": "You are a helpful assistant. Be helpful for code generation"},
                {"role": "user", "content" : instruction}
            ]
        if prompt:
            messages.insert(1, {
                "role": "user", "content" : prompt
            })

        payload = {
            "model" : model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        original_response = requests.post(url, json=payload, headers= self.headers).json()

        # Handle errors
        check_openai_errors(original_response)

        standardized_response = CodeGenerationDataClass(
            generated_text = original_response['choices'][0]['message']['content']
        )
        return ResponseType[CodeGenerationDataClass](
            original_response=original_response,
            standardized_response = standardized_response
        )

        
    def text__generation(
        self, text : str, 
        temperature : float, 
        max_tokens : int,
        model : str,
    ) -> ResponseType[GenerationDataClass]:
        url = f"{self.url}/completions"

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
        
    def text__custom_named_entity_recognition(
        self, 
        text: str,
        entities: List[str],
        examples: Optional[List[Dict]]= None) -> ResponseType[CustomNamedEntityRecognitionDataClass]:
        url = f"{self.url}/completions"
        built_entities = ','.join(entities)
        prompt = construct_custom_ner_instruction(text, built_entities, examples)
        payload = {
        "prompt" : prompt,
        "model" : self.model,
        "temperature" : 0.0,
        "top_p": 1,
        "max_tokens":250,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        
        # Handle errors
        if response.status_code != 200:
            raise ProviderException(response.text, response.status_code)
        
        try:
            original_response = response.json()
            items = json.loads(original_response['choices'][0]['text']) 
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException("An error occurred while parsing the response.") from exc
        
        standardized_response = CustomNamedEntityRecognitionDataClass(items=items.get('items'))

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
        
    def text__spell_check(self, text: str, language: str) -> ResponseType[SpellCheckDataClass]:
        url = f"{self.url}/completions"

        prompt = construct_spell_check_instruction(text, language)

        payload = {
            "n": 1,
            "model": "text-davinci-003",
            "max_tokens": 500,
            "temperature": 0.7,
            "prompt": prompt
        }

        try:
            original_response = requests.post(url, json=payload, headers=self.headers).json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error") from exc

        check_openai_errors(original_response)

        try:
            print(original_response['choices'][0]['text'])
            original_items = json.loads(original_response['choices'][0]['text'])
        except (KeyError, json.JSONDecodeError) as exc:
            print(exc)
            raise ProviderException("An error occurred while parsing the response.") from exc

        items: Sequence[SpellCheckItem] = []
        for item in original_items['items']:
            # The offset return by OpenAI aren't real offsets, so we need to found the real offset with the word and the approximate offset
            real_offset = text.find(item['text'], item['offset'])
            print(item['text'])
            print(item['offset'])
            print(real_offset)
            items.append(SpellCheckItem(
                text=item['text'],
                offset=real_offset,
                length=len(item['text']),
                type=item['type'],
                suggestions=item['suggestions'],
            ))

        return ResponseType[SpellCheckDataClass](
            original_response=original_response,
            standardized_response=SpellCheckDataClass(text=text, items=items)
        )

    def text__named_entity_recognition(self, language: str, text: str) -> ResponseType[NamedEntityRecognitionDataClass]:
        url = f"{self.url}/completions"

        prompt = construct_ner_instruction(text)

        payload = {
            "n": 1,
            "model": "text-davinci-003",
            "max_tokens": 500,
            "temperature": 0.7,
            "prompt": prompt
        }
        try:
            original_response = requests.post(url, json=payload, headers=self.headers).json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error") from exc

        check_openai_errors(original_response)

        try:
            original_items = json.loads(original_response['choices'][0]['text'])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProviderException("An error occurred while parsing the response.") from exc

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response=original_response,
            standardized_response=NamedEntityRecognitionDataClass(items=original_items['items'])
        )


    def text__embeddings(self, texts: List[str]) -> ResponseType[EmbeddingsDataClass]:
        url = 'https://api.openai.com/v1/embeddings'
        if len(texts) == 1:
            texts = texts[0]
        payload = {
            'input':texts,
            'model':'text-embedding-ada-002',
        }

        try:
            original_response = requests.post(url, json=payload, headers=self.headers).json()
        except json.JSONDecodeError as exc:
            raise ProviderException("Internal Server Error") from exc

        check_openai_errors(original_response)

        items : Sequence[EmbeddingsDataClass] = []
        embeddings = original_response['data']

        for embedding in embeddings:
            items.append(EmbeddingDataClass(embedding=embedding['embedding']))

        standardized_response = EmbeddingsDataClass(items=items)

        return ResponseType[EmbeddingsDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )

    def text__chat(
        self,
        text : str,
        chatbot_global_action: Optional[str],
        previous_history : Optional[List[Dict[str, str]]],
        temperature : float, 
        max_tokens : int,
        model : str,
        ) -> ResponseType[ChatDataClass]:
        url = f"{self.url}/chat/completions"
        
        messages = [{"role" : "user", "content" : text}]
    
        if previous_history:
            for idx, message in enumerate(previous_history):
                messages.insert(idx, {"role":message.get("role"), "content": message.get("message")})
        
        if chatbot_global_action:
            messages.insert(0, {
                "role": "system", "content" : chatbot_global_action
            })
            
        payload = {
            "model" : model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        original_response = requests.post(url, json=payload, headers= self.headers).json()

        # Handle errors
        check_openai_errors(original_response)
        
        # Standardize the response 
        generated_text = original_response['choices'][0]['message']['content']
        message = [
            ChatMessageDataClass(role='user', message = text),
            ChatMessageDataClass(role = 'assistant', message = generated_text)]
        
        standardized_response = ChatDataClass(
            generated_text = generated_text, 
            message = message
        )
        
        return ResponseType[ChatDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )