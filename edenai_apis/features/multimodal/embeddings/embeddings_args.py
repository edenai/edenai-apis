# pylint: disable=locally-disabled, line-too-long
import os
from typing import Dict, Any

from edenai_apis.utils.files import create_file_wrapper_for_sample

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")

image_path = f"{data_path}/logo_detection.jpeg"
video_path = f"{data_path}/logo.mp4"


def embeddings_arguments(provider_name: str) -> Dict[str, Any]:
    return {
        "text": "This is logos",
        "image": create_file_wrapper_for_sample(image_path),
        "video": create_file_wrapper_for_sample(video_path),
        "dimension": "xs",
    }
