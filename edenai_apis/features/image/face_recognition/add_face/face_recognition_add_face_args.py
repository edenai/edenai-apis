import mimetypes
import os
from typing import Dict

from pydub.utils import mediainfo

from edenai_apis.utils.files import FileInfo, FileWrapper
from ..common_args import COLLECTION_ID


def face_recognition_add_face_arguments(provider_name: str) -> Dict:
    feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    data_path = os.path.join(feature_path, "data")

    face_path = f"{data_path}/face_recognition_1.jpg"

    mime_type = mimetypes.guess_type(face_path)[0]
    file_info = FileInfo(
        os.stat(face_path).st_size,
        mime_type,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        mediainfo(face_path).get("sample_rate", "44100"),
        mediainfo(face_path).get("channels", "1"),
    )
    file_wrapper = FileWrapper(face_path, "", file_info)
    return {"file": file_wrapper, "collection_id": COLLECTION_ID}
