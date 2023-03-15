import base64
from io import BytesIO
import json
from typing import Literal
import requests
from edenai_apis.features.audio.text_to_speech.text_to_speech_dataclass import TextToSpeechDataClass
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.features.audio import AudioInterface
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.upload_s3 import USER_PROCESS, upload_file_bytes_to_s3

class LovoaiApi(ProviderInterface, AudioInterface):
    provider_name = 'lovoai'

    def __init__(self):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        self.url = self.api_settings['base_url']

        self.headers = {
            "apiKey": self.api_settings['api_key'],
            "Content-Type": "application/json"
        }

    availables_speakers = {
        "en-US": { "MALE": "Austin Hopkins", "FEMALE": "Susan Cole" },
        "en-GB": { "MALE": "Chad Taylor", "FEMALE": "Caroline Hughes" },
        "en-AU": { "MALE": "Kenny Marlowe", "FEMALE": "Rose Baker" },
        "fr-CA": { "MALE": "Antoine Mendy", "FEMALE": "Sylvie Minolet" },
        "es-AR": { "MALE": "Tomas Mondejar", "FEMALE": "Elena Mirabal" },
        "fr-FR": { "MALE": "Henri Malherbe", "FEMALE": "Denise Macon" },
        "de": { "MALE": "Conrad Martens", "FEMALE": "Katja Mahler" },
        "it": { "MALE": "Diego Manera", "FEMALE": "Elsa Micollo" },
        "pt-BR": { "MALE": "Antonio Munoz", "FEMALE": "Francisca Mesquita" },
        "pt": { "MALE": "Duarte Machado", "FEMALE": "Fernanda Maia" },
        "es": { "MALE": "Alonso Mairal", "FEMALE": "Paloma Maja" },
        "hu": { "MALE": "Csaba Nagy", "FEMALE": "Dorottya Varga" },
        "ja": { "MALE": "Genji Fukurama", "FEMALE": "Himari Honda" },
        "vi-VN": { "MALE": "Binh Pan", "FEMALE": "Hahn P." },
        "sv-SE": { "MALE": "", "FEMALE": "Ebba S." },
        "tr": { "MALE": "Derya O.", "FEMALE": "Hiranur B." },
        "uk": { "MALE": "", "FEMALE": "Olena H." },
        "sk": { "MALE": "", "FEMALE": "Jirina F." },
        "ru": { "MALE": "Ivan Chkalov", "FEMALE": "Lia Abakumov" },
        "pl": { "MALE": "Kacper F.", "FEMALE": "Maja L." },
        "nb-NO": { "MALE": "Aksel A.", "FEMALE": "Anita T." },
        "nn-NO": { "MALE": "Aksel A.", "FEMALE": "Anita T." },
        "zh-TW": { "MALE": "Fang Xi", "FEMALE": "Jia Qiao" },
        "zh-CN": { "MALE": "Yong Zheng", "FEMALE": "Mingzhu Bai" },
        "ko": { "MALE": "Jio Lee", "FEMALE": "Jisoo Paek" },
        "hi": { "MALE": "Chandran Dayal", "FEMALE": "Deepa Patel" },
        "el": { "MALE": "", "FEMALE": "Penelope A." },
        "fi": { "MALE": "", "FEMALE": "Valda M." },
        "en-PH": { "MALE": "", "FEMALE": "Angel G." },
        "en-IN": { "MALE": "Nikhil Patel", "FEMALE": "Kena Rao" },
        "nl": { "MALE": "Daan Bakker", "FEMALE": "Sjaan Van de Berg" },
        "da": { "MALE": "", "FEMALE": "Ditte J." },
        "cz": { "MALE": "", "FEMALE": "Jolana N." },
        "ar": { "MALE": "Ekram G.", "FEMALE": "Aaliyah Sarraf" },
    }

    @classmethod
    def get_speaker_id(cls, language: str, option: Literal["MALE", "FEMALE"]) -> str:
        speaker_id = cls.availables_speakers[language][option]
        print(speaker_id)
        if speaker_id == "":
            raise ProviderException(f"Speaker {option} for language {language} is not available")
        return speaker_id

    def audio__text_to_speech(
        self,
        language: str,
        text: str,
        option: Literal["MALE", "FEMALE"]
    ) -> ResponseType[TextToSpeechDataClass]:
        data = json.dumps({
            "text": text,
            "speaker_id": LovoaiApi.get_speaker_id(language, option),
        })
        print(data)

        response = requests.post(f'{self.url}v1/conversion', headers=self.headers, data=data)

        if response.status_code != 200:
            raise ProviderException(response.json().get('error', "Something went wrong"))

        audio_content = BytesIO(response.content)
        audio = base64.b64encode(audio_content.read()).decode("utf-8")

        audio_content.seek(0)
        resource_url = upload_file_bytes_to_s3(audio_content, ".wav", USER_PROCESS)

        return ResponseType[TextToSpeechDataClass](
            original_response=response,
            standardized_response=TextToSpeechDataClass(
                audio=audio, 
                voice_type=0,
                audio_resource_url = resource_url
            )
        )
