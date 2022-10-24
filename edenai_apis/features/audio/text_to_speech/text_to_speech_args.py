from typing import Dict
import os

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def text_to_speech_arguments() -> Dict:
    return {
        "language": "fr-FR",
        "text": "Bonjour Je m'appelle Jane",
        "option": "FEMALE",
    }
