import mimetypes
import os

from pydub.utils import mediainfo

from edenai_apis.utils.files import FileInfo, FileWrapper


def document_translation_arguments(provider_name: str) -> dict:
    feature_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(feature_path, "data")

    document_path = f"{data_path}/document_translation.pdf"

    mime_type = mimetypes.guess_type(document_path)[0]
    file_info = FileInfo(
        os.stat(document_path).st_size,
        mime_type,
        [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        mediainfo(document_path).get("sample_rate", "44100"),
        mediainfo(document_path).get("channels", "1"),
    )
    file_wrapper = FileWrapper(document_path, "", file_info)
    return {
        "file": file_wrapper,
        "file_type": file_info.file_media_type,
        "source_language": "en",
        "target_language": "fr",
    }
