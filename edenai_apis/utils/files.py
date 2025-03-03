import os
from typing import Optional, List


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

    file_size: int
    file_media_type: str
    supported_extensions: List[str]
    # for audio and video files
    file_frame_rate: Optional[str]
    file_channels: Optional[str]
    file_duration: Optional[float]


class FileWrapper:
    def __init__(self, file_path, file_url, file_info) -> None:
        self.file_path = file_path
        self.file_url = file_url
        self.file_info = file_info

    file_path: Optional[str]
    file_url: Optional[str]
    file_info: FileInfo

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
