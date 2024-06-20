from typing import Optional, Sequence
from pydantic import BaseModel, Field, StrictStr


class LogoVertice(BaseModel):
    x: Optional[float] = Field(description="The x-coordinate of the vertex.")
    y: Optional[float] = Field(description="The y-coordinate of the vertex.")


class LogoBoundingPoly(BaseModel):
    vertices: Optional[Sequence[LogoVertice]] = Field(
        default_factory=list, description="Vertices of the logos in the image"
    )


class LogoItem(BaseModel):
    bounding_poly: LogoBoundingPoly = LogoBoundingPoly()
    description: Optional[StrictStr] = Field(description="Name of the logo")
    score: Optional[float] = Field(
        description="Confidence score how sure it's this is a real logo."
    )


class LogoDetectionDataClass(BaseModel):
    items: Sequence[LogoItem] = Field(
        default_factory=list,
        description="List of the detected brands logo from the image.",
    )
