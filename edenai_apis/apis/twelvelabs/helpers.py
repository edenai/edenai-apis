import requests
from typing import Dict
from edenai_apis.features.video.logo_detection_async.logo_detection_async_dataclass import (
    LogoDetectionAsyncDataClass,
    LogoTrack,
    VideoLogo,
    VideoLogoBoundingBox,
)
import requests
from edenai_apis.features.video.text_detection_async.text_detection_async_dataclass import (
    TextDetectionAsyncDataClass,
    VideoText,
    VideoTextFrames,
    VideoTextBoundingBox,
)


def convert_json_to_logo_dataclass(data: Dict) -> LogoDetectionAsyncDataClass:
    try:
        logos_tracks = []
        for item in data["data"]:
            timestamp = float(item["end"]) - float(item["start"])
            video_logo = VideoLogo(
                timestamp=timestamp,
                bounding_box=VideoLogoBoundingBox(
                    top=None, left=None, height=None, width=None
                ),
                confidence=item.get("score"),
            )
            logo_track = LogoTrack(description=item["value"], tracking=[video_logo])
            logos_tracks.append(logo_track)

        return LogoDetectionAsyncDataClass(logos=logos_tracks)
    except Exception as e:
        return LogoDetectionAsyncDataClass(logos=[])


def convert_json_to_text_dataclass(data: Dict) -> TextDetectionAsyncDataClass:
    try:
        texts = []
        for item in data["data"]:
            video_text_frame = VideoTextFrames(
                confidence=1.0,
                timestamp=float(item["end"]) - float(item["start"]),
                bounding_box=VideoTextBoundingBox(
                    top=None, left=None, height=None, width=None
                ),
            )
            video_text = VideoText(text=item["value"], frames=[video_text_frame])
            texts.append(video_text)

        return TextDetectionAsyncDataClass(texts=texts)
    except Exception as e:
        return TextDetectionAsyncDataClass(texts=[])
