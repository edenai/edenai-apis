from typing import Dict

from watson_developer_cloud import NaturalLanguageUnderstandingV1, LanguageTranslatorV3
from watson_developer_cloud.speech_to_text_v1 import SpeechToTextV1
from watson_developer_cloud.text_to_speech_v1 import TextToSpeechV1


def ibm_clients(api_settings: Dict) -> Dict:
    return {
        "text": NaturalLanguageUnderstandingV1(
            version="2021-08-01",
            iam_apikey=api_settings["natural_language_understanding"]["apikey"],
            url=api_settings["natural_language_understanding"]["url"],
        ),
        "texttospeech": TextToSpeechV1(
            iam_apikey=api_settings["text_to_speech"]["apikey"],
            url=api_settings["text_to_speech"]["url"],
        ),
        "translation": LanguageTranslatorV3(
            version="2018-05-01",
            iam_apikey=api_settings["translator"]["apikey"],
            url=api_settings["translator"]["url"],
        ),
        "speech": SpeechToTextV1(
            iam_apikey=api_settings["speech_to_text"]["apikey"],
            url=api_settings["speech_to_text"]["url"],
        ),
    }


models = {
    "fr-FR": "fr-FR_NarrowbandModel",
    "en-US": "en-US_NarrowbandModel",
    "en-GB": "en-GB_NarrowbandModel",
    "es-ES": "es-ES_NarrowbandModel",
    "it-IT": "it-IT_NarrowbandModel",
    "ja-JP": "ja-JP_NarrowbandModel",
    "nl-NL": "nl-NL_NarrowbandModel",
    "pt-PT": "pt-BR_NarrowbandModel",
    "de-DE": "de-DE_NarrowbandModel",
}

audio_voices_ids = {
    "en-US": {"FEMALE": "en-US_AllisonExpressive", "MALE": "en-US_MichaelExpressive"},
    "fr-FR": {"FEMALE": "fr-FR_ReneeV3Voice", "MALE": "fr-FR_NicolasV3Voice"},
    "es-ES": {"FEMALE": "es-ES_LauraV3Voice", "MALE": "es-ES_EnriqueV3Voice"},
    "de-DE": {"FEMALE": "de-DE_BirgitV3Voice", "MALE": "de-DE_DieterV3Voice"},
    "en-GB": {"FEMALE": "en-GB_KateV3Voice", "MALE": "en-GB_JamesV3Voice"},
    "it-IT": {"FEMALE": "it-IT_FrancescaV3Voice", "MALE": ""},
    "ja-JP": {"FEMALE": "ja-JP_EmiV3Voice", "MALE": ""},
    "pt-BR": {"FEMALE": "pt-BR_IsabelaV3Voice", "MALE": ""},
    "fr-CA": {"FEMALE": "fr-CA_LouiseV3Voice", "MALE": ""},
    "es-LA": {"FEMALE": "es-LA_SofiaV3Voice", "MALE": ""},
    "es-US": {"FEMALE": "es-US_SofiaV3Voice", "MALE": ""},
    "en-AU": {"FEMALE": "en-AU_HeidiExpressive", "MALE": "en-AU_JackExpressive"},
    "nl-NL": {"FEMALE": "nl-NL_MerelV3Voice"},
    "ko-KR": {"FEMALE": "ko-KR_JinV3Voice"},
}

language_iso = {
    "fr-FR": "fr",
    "en-US": "en",
    "es-ES": "es",
    "fn-FN": "fi",
    "sw-SW": "sv",
}

tags = {
    "ADJ": "Adjactive",
    "ADP": "Adposition",
    "ADV": "Adverb",
    "AUX": "Auxiliary",
    "CCONJ": "Coordinating_Conjunction",
    "INTJ": "Injection",
    "DET": "Determiner",
    "NOUN": "Noun",
    "NUM": "Cardinal number",
    "PRON": "Pronoun",
    "PROPN": "Proper noun",
    "PART": "Particle",
    "PUNCT": "Punctuation",
    "SYM": "Symbol",
    "SCONJ": "Sub Conjunction",
    "VERB": "Verb",
    "X": "Other",
}
