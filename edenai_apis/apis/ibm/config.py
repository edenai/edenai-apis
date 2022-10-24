from watson_developer_cloud import NaturalLanguageUnderstandingV1, LanguageTranslatorV3
from watson_developer_cloud.text_to_speech_v1 import TextToSpeechV1
from watson_developer_cloud.speech_to_text_v1 import SpeechToTextV1

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


api_settings = load_provider(ProviderDataEnum.KEY, "ibm")

clients = {
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
    "en-US": {"FEMALE": "en-US_AllisonVoice", "MALE": "en-US_MichaelVoice"},
    "fr-FR": {"FEMALE": "fr-FR_ReneeVoice", "MALE": "fr-FR_NicolasV3Voice"},
    "es-ES": {"FEMALE": "es-ES_LauraVoice", "MALE": "es-ES_EnriqueVoice"},
    "ar-XA": {"FEMALE": "", "MALE": "ar-AR_OmarVoice"},
    "de-DE": {"FEMALE": "de-DE_BirgitVoice", "MALE": "de-DE_DieterVoice"},
    "en-GB": {"FEMALE": "en-GB_KateVoice", "MALE": "en-GB_JamesV3Voice"},
    "nl-NL": {"FEMALE": "nl-NL_EmmaVoice", "MALE": "nl-NL_LiamVoice"},
    "it-IT": {"FEMALE": "it-IT_FrancescaVoice", "MALE": ""},
    "ja-JP": {"FEMALE": "ja-JP_EmiVoice", "MALE": ""},
    "cmn-CN": {"FEMALE": "zh-CN_LiNaVoice", "MALE": "zh-CN_WangWeiVoice"},
    "ru-RU": {"FEMALE": "", "MALE": ""},
    "pt-BR": {"FEMALE": "pt-BR_IsabelaVoice", "MALE": ""},
    "da-DK": {"FEMALE": "", "MALE": ""},
    "ko-KR": {"FEMALE": "ko-KR_YoungmiVoice", "MALE": ""},
    "pt-PT": {"FEMALE": "", "MALE": ""},
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
