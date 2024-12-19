# pylint: disable=locally-disabled, line-too-long
import mimetypes
import os
from typing import Dict, Any

from pydub.utils import mediainfo

from edenai_apis.utils.files import FileInfo, FileWrapper


def question_answer_arguments(provider_name: str) -> Dict[str, Any]:
    feature_path = os.path.dirname(os.path.dirname(__file__))

    data_path = os.path.join(feature_path, "data")

    image_path = f"{data_path}/logo_detection.jpeg"

    mime_type = mimetypes.guess_type(image_path)[0]
    file_info = FileInfo(
        os.stat(image_path).st_size,
        mime_type,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        mediainfo(image_path).get("sample_rate", "44100"),
        mediainfo(image_path).get("channels", "1"),
    )
    file_wrapper = FileWrapper(image_path, "", file_info)
    return {
        "file": file_wrapper,
        "question": "What are the logos on the image ?",
        "temperature": 0.0,
        "max_tokens": 64,
        "settings": {
            "alephalpha": "luminous-extended",
            "google": "gemini-1.5-flash",
            "openai": "gpt-4o",
        },
    }
