from typing import Literal, Optional

from pydantic import Field

from edenai_apis.features.image.background_removal.types import BackgroundRemovalParams


class PhotoroomBackgroundRemovalParams(BackgroundRemovalParams):
    format: Literal["jpg", "png"] = Field(
        "png", description="Output format of the image."
    )
    channel: Literal["rgba", "alpha"] = Field(
        "rgba", description="Output channel of the image."
    )
    bg_color: Optional[str] = Field(
        None,
        description="Background color of the image."
        + "If not specified, the background color will be transparent."
        + "Can be a hex code (#FF00FF) or a HTML color (red, green, etc.)",
    )
    size: Literal["preview", "medium", "hd", "full"] = Field(
        "full", description="Size of the output image."
    )
    crop: bool = Field(
        False,
        description="If true, the image returned is cropped to the cutout border."
        + "Transparent pixels are removed from the borders.",
    )
