from typing import Dict
import os

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def face_detection_arguments() -> Dict:
    return {"file": open(f"{data_path}/faces.mp4", "rb")}
