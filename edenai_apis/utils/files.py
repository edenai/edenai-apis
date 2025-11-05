import os
import tempfile
from typing import List, Optional


class FileInfo:
    def __init__(
        self, file_size, file_mimetype, file_extension, *args, **kwargs
    ) -> None:
        self.file_size = file_size
        self.file_media_type = file_mimetype
        self.file_extension = file_extension
        if args:
            self.file_frame_rate, self.file_channels = args
        self.file_duration = kwargs.get("duration", None)
        self.metadata = {**kwargs}

    file_size: int
    file_media_type: str
    supported_extensions: List[str]
    # for audio and video files
    file_frame_rate: Optional[str]
    file_channels: Optional[str]
    file_duration: Optional[float]


class FileWrapper:
    def __init__(
        self,
        file_path,
        file_url,
        file_info,
        file_b64_content: str = None,
    ) -> None:
        self.file_path = file_path
        self.file_url = file_url
        self.file_info = file_info
        self._file_b64_content = file_b64_content

    file_path: Optional[str]
    file_url: Optional[str]
    file_info: FileInfo

    _file_b64_content: Optional[str] = None

    def get_file_b64_content(self):
        if self._file_b64_content:
            return self._file_b64_content
        if self.file_path:
            with tempfile.NamedTemporaryFile("rb") as f:
                self._file_b64_content = f.read()
                f.close()
                return self._file_b64_content
        raise Exception("No file found...!")

    def get_file_content(self):
        if self.file_url:
            return self.file_url
        if self.file_path:
            return self.file_path
        raise Exception("No file found...!")

    def close_file(self):
        if not self.file_path:
            return
        try:
            os.remove(self.file_path)
        except OSError as e:
            # The file was moved or deleted before the tempfile could unlink
            pass
