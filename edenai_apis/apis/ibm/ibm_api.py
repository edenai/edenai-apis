from io import BufferedReader

import base64

from typing import Sequence

import json

from ibm_watson.natural_language_understanding_v1 import (
    SentimentOptions,
    Features,
    KeywordsOptions,
    EntitiesOptions,
    SyntaxOptions,
    SyntaxOptionsTokens,
)
from edenai_apis.features.audio import (
    SpeechToTextAsyncDataClass,
    TextToSpeechDataClass,
    SpeechDiarizationEntry,
    SpeechDiarization
)

from edenai_apis.features.text import(
    InfosKeywordExtractionDataClass,
    KeywordExtractionDataClass,
    InfosNamedEntityRecognitionDataClass,
    NamedEntityRecognitionDataClass,
    SentimentAnalysisDataClass,
    InfosSyntaxAnalysisDataClass,
    SyntaxAnalysisDataClass,
    SegmentSentimentAnalysisDataClass,
    SentimentEnum,
)

from edenai_apis.features.translation import (
    AutomaticTranslationDataClass,
    InfosLanguageDetectionDataClass,
    LanguageDetectionDataClass
)

from edenai_apis.utils.audio import file_with_good_extension
from edenai_apis.utils.exception import ProviderException, LanguageException

from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType
    )
from edenai_apis.features import ProviderApi, Translation, Audio, Text

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from .config import ibm_clients, audio_voices_ids, tags

from watson_developer_cloud.watson_service import WatsonApiException

class IbmApi(
    ProviderApi,
    Translation,
    Audio,
    Text,
):

    provider_name = "ibm"
    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, "ibm")
        self.clients = ibm_clients(self.api_settings)


    def translation__automatic_translation(
        self, source_language: str,
        target_language: str,
        text: str
    ) -> ResponseType[AutomaticTranslationDataClass]:
        # Getting response of API

        response = (
            self.clients["translation"]
            .translate(text=text, source=source_language, target=target_language)
            .get_result()
        )

        # Create output TextAutomaticTranslation object
        standarized: AutomaticTranslationDataClass

        # Getting the translated text
        for translated_text in response["translations"]:
            standarized = AutomaticTranslationDataClass(
                text=translated_text["translation"]
            )

        return ResponseType[AutomaticTranslationDataClass](
            original_response= response["translations"],
            standarized_response = standarized
        )


    def translation__language_detection(self,
            text: str
    ) -> ResponseType[LanguageDetectionDataClass]:
        """
        :param text:        String that contains input text
        :return:            String that contains output result
        """

        response = self.clients["translation"].identify(text).get_result()

        # Getting the language's code detected and its score of confidence
        items: Sequence[InfosLanguageDetectionDataClass] = []

        if len(response["languages"]) > 0:
            for lang in response["languages"]:
                if lang["confidence"] > 0.2:
                    items.append(
                        InfosLanguageDetectionDataClass(
                            language=lang["language"], confidence=lang["confidence"]
                        )
                    )

        standarized_response = LanguageDetectionDataClass(items=items)

        return ResponseType[LanguageDetectionDataClass] (
            original_response= response,
            standarized_response = standarized_response
        )


    def text__sentiment_analysis(self,
        language: str,
        text: str
    ) -> ResponseType[SentimentAnalysisDataClass]:
        try:
            response = (
                self.clients["text"]
                .analyze(
                    text=text,
                    language=language,
                    features=Features(sentiment=SentimentOptions()),
                )
                .get_result()
            )
        except WatsonApiException as exc:
            if "not enough text for language id" in exc.message:
                raise LanguageException(exc.message)
        # Create output object
        items: Sequence[SegmentSentimentAnalysisDataClass] = []
        standarize = SentimentAnalysisDataClass(
            general_sentiment=response["sentiment"]["document"]["label"],
            general_sentiment_rate=float(
                    abs(response["sentiment"]["document"]["score"])
                ),
            items=items,
        )

        return ResponseType[SentimentAnalysisDataClass](
            original_response = response,
            standarized_response = standarize
        )


    def audio__text_to_speech(self,
        language: str,
        text: str,
        option: str
    ) -> ResponseType[TextToSpeechDataClass]:
        """
        :param language:    String that contains language name 'fr-FR', 'en-US', 'es-EN'
        :param text:        String that contains text to transform
        :param option:      String that contains option of voice(MALE, FEMALE)
        :return:
        """

        # Formatting (option, language) to voice id supported by IBM API
        voiceid = audio_voices_ids[language][option]

        response = (
            self.clients["texttospeech"]
            .synthesize(text=text, accept="audio/mp3", voice=voiceid)
            .get_result()
        )

        audio = base64.b64encode(response.content).decode("utf-8")
        voice_type = 1

        standarized_response = TextToSpeechDataClass(audio=audio, voice_type=voice_type)

        return ResponseType[TextToSpeechDataClass](
            original_response = {},
            standarized_response = standarized_response
        )


    def text__keyword_extraction(self,
        language: str,
        text: str
    ) -> ResponseType[KeywordExtractionDataClass]:
        """
        :param language:    String that contains language name 'fr-FR', 'en-US', 'es-EN'
        :param text:        String that contains input text
        :return:
        """
        try:
            response = (
                self.clients["text"]
                .analyze(
                    text=text,
                    language=language,
                    features=Features(
                        keywords=KeywordsOptions(emotion=True, sentiment=True)
                    ),
                )
                .get_result()
            )
        except WatsonApiException as exc:
            if "not enough text for language id" in exc.message:
                raise LanguageException(exc.message)

        # Analysing response
        items: Sequence[InfosKeywordExtractionDataClass] = []
        for key_phrase in response["keywords"]:
            items.append(
                InfosKeywordExtractionDataClass(
                    keyword=key_phrase["text"], importance=key_phrase["relevance"]
                )
            )

        standarized_response = KeywordExtractionDataClass(items=items)

        return ResponseType[KeywordExtractionDataClass](
            original_response = response,
            standarized_response = standarized_response
        )



    def text__named_entity_recognition(self,
        language:str,
        text:str
    ) -> ResponseType[NamedEntityRecognitionDataClass]:

        try:
            response = (
                self.clients["text"]
                .analyze(
                    text=text,
                    language=language,
                    features=Features(
                        entities=EntitiesOptions(
                            sentiment=True, mentions=True, emotion=True
                        )
                    ),
                )
                .get_result()
            )
        except WatsonApiException as exc:
            if "not enough text for language id" in exc.message:
                raise LanguageException(exc.message)

        items: Sequence[InfosNamedEntityRecognitionDataClass] = []

        for ent in response["entities"]:
            category = ent["type"].upper()
            if category == 'JOBTITLE':
                category = 'PERSONTYPE'
            items.append(
                InfosNamedEntityRecognitionDataClass(
                    entity=ent["text"],
                    importance=ent["relevance"],
                    category=category,
                )
            )

        standarized_response = NamedEntityRecognitionDataClass(items=items)

        return ResponseType[NamedEntityRecognitionDataClass](
            original_response = response,
            standarized_response = standarized_response
        )


    def text__syntax_analysis(self,
        language: str,
        text: str
    ) -> ResponseType[SyntaxAnalysisDataClass]:
        """
        :param language:    String that contains language name 'fr-FR', 'en-US', 'es-EN'
        :param text:        String that contains input text
        :return:            Array containing api response and TextSyntaxAnalysis Object that
        contains the sentiments and their syntax
        """

        try:
            response = (
                self.clients["text"]
                .analyze(
                    text=text,
                    language=language,
                    features=Features(
                        syntax=SyntaxOptions(
                            sentences=True,
                            tokens=SyntaxOptionsTokens(lemma=True, part_of_speech=True),
                        )
                    ),
                )
                .get_result()
            )
        except WatsonApiException as exc:
            if "not enough text for language id" in exc.message:
                raise LanguageException(exc.message)

        items: Sequence[InfosSyntaxAnalysisDataClass] = []

        # Getting syntax detected of word and its score of confidence
        for keyword in response["syntax"]["tokens"]:
            tag_ = tags[keyword["part_of_speech"]]
            if "lemma" in keyword:
                items.append(
                    InfosSyntaxAnalysisDataClass(
                        word=keyword["text"],
                        tag=tag_,
                        lemma=keyword["lemma"],
                    )
                )
            else:
                items.append(
                    InfosSyntaxAnalysisDataClass(
                        word=keyword["text"],
                        tag=tag_,
                    )
                )

        standarized_response = SyntaxAnalysisDataClass(items=items)

        return ResponseType[SyntaxAnalysisDataClass](
            original_response = response,
            standarized_response = standarized_response
        )


    def audio__speech_to_text_async__launch_job(
        self,
        file: BufferedReader,
        language: str, speakers : int, profanity_filter: bool,
        vocabulary: list
    ) -> AsyncLaunchJobResponseType:

        # check if audio file needs convertion
        accepted_extensions = ["flac", "mp3", "wav", "flac", "ogg", "webm", "alaw", "amr", 
        "g729", "l16", "mpeg", "mulaw"]
        new_file, export_format, channels, frame_rate = file_with_good_extension(file, accepted_extensions)

        language_audio = language
        audio_config = {
            "audio" : new_file,
            "content_type" : "audio/"+export_format,
            "speaker_labels" : True,
            "profanity_filter" : profanity_filter,
        }
        audio_config.update({
            "rate": frame_rate
        })
        if language_audio:
            audio_config.update({
                "model" : f"{language_audio}_NarrowbandModel"
            })
        response = self.clients["speech"].create_job(**audio_config)
        if response.status_code == 201:
            return AsyncLaunchJobResponseType(
                provider_job_id=response.result["id"]
            )
        else:
            raise ProviderException("An error occured during ibm api call")

    def audio__speech_to_text_async__get_job_result(
        self,
        provider_job_id: str
    ) -> AsyncBaseResponseType[SpeechToTextAsyncDataClass]:
        response = self.clients["speech"].check_job(provider_job_id)
        status = response.result["status"]
        if status == "completed":
            original_response = response.result["results"]
            data = response.result["results"][0]["results"]

            diarization_entries = []
            speakers = set()

            text = " ".join([entry["alternatives"][0]["transcript"] for entry in data])

            time_stamps = [time_stamp for entry in data for time_stamp in entry["alternatives"][0]["timestamps"]]
            for idx_word, word_info in enumerate(original_response[0]["speaker_labels"]):
                speakers.add(word_info["speaker"])
                diarization_entries.append(
                    SpeechDiarizationEntry(
                        segment= time_stamps[idx_word][0],
                        start_time= str(time_stamps[idx_word][1]),
                        end_time= str(time_stamps[idx_word][2]),
                        speaker= word_info["speaker"] + 1,
                        confidence= word_info["confidence"]
                    )
                )
            diarization = SpeechDiarization(total_speakers=len(speakers), entries= diarization_entries)
            standarized_response = SpeechToTextAsyncDataClass(text=text, diarization= diarization)
            return AsyncResponseType[SpeechToTextAsyncDataClass](
                original_response = original_response,
                standarized_response = standarized_response,
                provider_job_id = provider_job_id
            )

        if status == "failed":
            # Apparently no error message present in response
            # ref: https://cloud.ibm.com/apidocs/speech-to-text?code=python#checkjob
            raise ProviderException

        return AsyncPendingResponseType[SpeechToTextAsyncDataClass](
            provider_job_id=provider_job_id
        )
