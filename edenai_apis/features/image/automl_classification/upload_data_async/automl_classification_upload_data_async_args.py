import mimetypes
import os
from typing import Dict

from pydub.utils import mediainfo

from edenai_apis.utils.files import FileInfo, FileWrapper


def automl_classification_upload_data_async_arguments(provider_name: str) -> Dict:
    feature_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(feature_path, "data")

    image_path = f"{data_path}/automl_classification.png"

    mime_type = mimetypes.guess_type(image_path)[0]
    file_info = FileInfo(
        os.stat(image_path).st_size,
        mime_type,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        mediainfo(image_path).get("sample_rate", "44100"),
        mediainfo(image_path).get("channels", "1"),
    )
    file_wrapper = FileWrapper(image_path, "", file_info)
    if provider_name == "nyckel":
        project_id = "function_v7nmvjzb7a7i80wg"
    else:
        raise NotImplementedError(
            f"Please add a project id for test arguments of provider: {provider_name}"
        )

    return {
        "project_id": project_id,
        "label": "Cat",
        "type_of_data": "TRAINING",
        "file": file_wrapper,
    }
