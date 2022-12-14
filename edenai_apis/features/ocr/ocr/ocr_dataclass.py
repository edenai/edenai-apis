from typing import Dict, Optional, Sequence

from pydantic import BaseModel, Field, StrictStr


class Bounding_box(BaseModel):
    text: Optional[StrictStr]
    left: Optional[float]
    top: Optional[float]
    width: Optional[float]
    height: Optional[float]


class OcrDataClass(BaseModel):
    text: StrictStr
    bounding_boxes: Sequence[Bounding_box] = Field(default_factory=list)

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["text"]
