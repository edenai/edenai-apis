import boto3
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

api_settings = load_provider(ProviderDataEnum.KEY, "amazon")

clients = {
    "speech": boto3.client(
        "transcribe",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "texttospeech": boto3.client(
        "polly",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "image": boto3.client(
        "rekognition",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "textract": boto3.client(
        "textract",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "text": boto3.client(
        "comprehend",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "translate": boto3.client(
        "translate",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "video": boto3.client(
        "rekognition",
        region_name=api_settings["video-region"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "text_classification": boto3.client(
        "sts",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "s3": boto3.client(
        "s3",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
}
storage_clients = {
    "speech": boto3.resource(
        "s3",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "textract": boto3.resource(
        "s3",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "text_classification": boto3.resource(
        "s3",
        region_name=api_settings["region_name"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
    "image": None,
    "text": None,
    "video": boto3.resource(
        "s3",
        region_name=api_settings["video-region"],
        aws_access_key_id=api_settings["aws_access_key_id"],
        aws_secret_access_key=api_settings["aws_secret_access_key"],
    ),
}

audio_voices_ids = {
    "en-US": {"FEMALE": "Kendra", "MALE": "Joey"},
    "fr-FR": {"FEMALE": "Celine", "MALE": "Mathieu"},
    "es-ES": {"FEMALE": "Lucia", "MALE": "Enrique"},
    "ar-XA": {"FEMALE": "Zeina", "MALE": ""},
    "de-DE": {"FEMALE": "Marlene", "MALE": "Hans"},
    "en-GB": {"FEMALE": "Amy", "MALE": "Brian"},
    "nl-NL": {"FEMALE": "Lotte", "MALE": "Ruben"},
    "it-IT": {"FEMALE": "Carla", "MALE": "Giorgio"},
    "ja-JP": {"FEMALE": "Mizuki", "MALE": "Takumi"},
    "cmn-CN": {"FEMALE": "Zhiyu", "MALE": ""},
    "ru-RU": {"FEMALE": "Tatyana", "MALE": "Maxim"},
    "pt-BR": {"FEMALE": "Camila", "MALE": "Ricardo"},
    "da-DK": {"FEMALE": "Naja", "MALE": "Mads"},
    "ko-KR": {"FEMALE": "Seoyeon", "MALE": ""},
    "pt-PT": {"FEMALE": "Ines", "MALE": "Cristiano"},
}

tags = {
    "ADJ": "Adjactive",
    "ADP": "Adposition",
    "ADV": "Adverb",
    "AUX": "Auxiliary",
    "CCONJ": "Coordinating_Conjunction",
    "CONJ": "Coordinating_Conjunction",
    "INTJ": "Injection",
    "DET": "Determiner",
    "NOUN": "Noun",
    "NUM": "Cardinal_number",
    "PRON": "Pronoun",
    "PROPN": "Proper_noun",
    "PART": "Particle",
    "PUNCT": "Punctuation",
    "SYM": "Symbol",
    "SCONJ": "Subordinating_Conjunction",
    "VERB": "Verb",
    "O": "Other",
}
