from typing import Literal

from pydantic import Field

from edenai_apis.features.image.background_removal.types import BackgroundRemovalParams


class MicrosoftBackgroundRemovalParams(BackgroundRemovalParams):
    mode: Literal["backgroundRemoval", "foregroundMatting"] = Field(
        "backgroundRemoval",
        description="The mode of the API. Default value is backgroundRemoval",
    )
