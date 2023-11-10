from typing import Literal

from pydantic import Field


class MicrosoftBackgroundRemovalParams:
    mode: Literal["backgroundRemoval", "foregroundMatting"] = Field(
        "backgroundRemoval",
        description="The mode of the API. Default value is backgroundRemoval",
    )
