from typing import Literal

from pydantic import Field, BaseModel


class MicrosoftBackgroundRemovalParams(BaseModel):
    mode: Literal["backgroundRemoval", "foregroundMatting"] = Field(
        "backgroundRemoval",
        description="The mode of the API. Default value is backgroundRemoval",
    )
