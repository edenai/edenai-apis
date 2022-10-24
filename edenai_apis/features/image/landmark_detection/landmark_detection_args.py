import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def landmark_detection_arguments() -> Dict:
    return {"file": open(f"{data_path}/landmark.jpeg", "rb")}
