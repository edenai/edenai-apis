import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def text_to_speech_arguments(provider_name: str) -> Dict:
    return {
        "language": "en",
        "text": "Hello my name is Jane",
        "option": "FEMALE",
        "settings": {},
        "audio_format": "",
        "speaking_rate": 0,
        "speaking_pitch": 0,
        "speaking_volume": 0,
        "sampling_rate": 0,
    }
