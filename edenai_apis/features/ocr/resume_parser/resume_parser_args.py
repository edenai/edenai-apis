import mimetypes
import os
from typing import Dict

from pydub.utils import mediainfo

from edenai_apis.utils.files import FileInfo, FileWrapper

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")

ocr_path = f"{data_path}/resume.pdf"


def resume_parser_arguments(provider_name: str) -> Dict:

    mime_type = mimetypes.guess_type(ocr_path)[0]
    file_info = FileInfo(
        os.stat(ocr_path).st_size,
        mime_type,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        mediainfo(ocr_path).get("sample_rate", "44100"),
        mediainfo(ocr_path).get("channels", "1"),
    )
    file_wrapper = FileWrapper(ocr_path, "", file_info)
    return {"file": file_wrapper, "settings": {"openai": "gpt-4o"}}
