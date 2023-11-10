from typing import Literal

from pydantic import Field


class Api4aiBackgroundRemovalParams:
    """Parameters for the background removal feature of the Api4ai API.

    Attributes:
        mode (Literal["fg-image", "fg-image-shadow", "fg-mask"], optional):
            The mode of the output image. Defaults to "fg-image".
    """

    mode: Literal["fg-image", "fg-image-shadow", "fg-mask"] = Field(
        "fg-image", description="The mode of the output image"
    )
