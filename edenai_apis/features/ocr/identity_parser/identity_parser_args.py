import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def identity_parser_arguments() -> Dict:
    # filename = 'passport-GB.jpg'
    # filename = 'card-ID_FR.jpeg'
    filename = 'passport-US.pdf'
    # filename = 'card-ID-UK.png'
    return {"file": open(f"{data_path}/{filename}", "rb")}