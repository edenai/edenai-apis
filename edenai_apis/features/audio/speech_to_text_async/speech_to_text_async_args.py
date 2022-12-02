from typing import Dict
import os

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def speech_to_text_arguments() -> Dict:
    return {
        "file": open(f"{data_path}/conversation.mp3", 
        "rb"), "language": "en", 
        "speakers" : 2, 
        "profanity_filter": False,
        "vocabulary" : []
        }
