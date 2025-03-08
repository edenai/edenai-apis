import mimetypes
import os
from typing import Dict

from pydub.utils import mediainfo

from edenai_apis.utils.files import FileInfo, FileWrapper


def face_compare_arguments(provider_name: str) -> Dict:
    feature_path = os.path.dirname(os.path.dirname(__file__))

    data_path = os.path.join(feature_path, "data")

    image1_path = f"{data_path}/face1.jpg"
    image2_path = f"{data_path}/face2.jpg"

    mime_type_1 = mimetypes.guess_type(image1_path)[0]
    mime_type_2 = mimetypes.guess_type(image2_path)[0]
    file_info_1 = FileInfo(
        os.stat(image1_path).st_size,
        mime_type_1,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type_1)],
        mediainfo(image1_path).get("sample_rate", "44100"),
        mediainfo(image1_path).get("channels", "1"),
    )
    file_info_2 = FileInfo(
        os.stat(image2_path).st_size,
        mime_type_2,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type_2)],
        mediainfo(image2_path).get("sample_rate", "44100"),
        mediainfo(image2_path).get("channels", "1"),
    )
    file_wrapper_1 = FileWrapper(image1_path, "", file_info_1)
    file_wrapper_2 = FileWrapper(image2_path, "", file_info_2)
    return {"file1": file_wrapper_1, "file2": file_wrapper_2}
