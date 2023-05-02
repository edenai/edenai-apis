import enum
from typing import Optional, Sequence
from pydantic import BaseModel, Field, validator

class BoundingBoxEnum(enum.Enum):
    """Enum for the keys of the bounding box JSON object
    
    Attributes:
        LEFT (int): Left coordinate of the bounding box
        TOP (int): Top coordinate of the bounding box
        WIDTH (int): Width of the bounding box
        HEIGHT (int): Height of the bounding box
    """
    LEFT = 0
    TOP = 1
    WIDTH = 2
    HEIGHT = 3

class BoundingBox(BaseModel):
    """Bounding box of a word in the image
    
    Attributes:
        left (float): Left coordinate of the bounding box
        top (float): Top coordinate of the bounding box
        width (float): Width of the bounding box
        height (float): Height of the bounding box
        text (str): Text detected in the bounding box
    """
    left: float
    top: float
    width: float
    height: float

    @staticmethod
    def from_json(
        bounding_box: dict,
        modifiers: callable = lambda x: x,
        keys: Sequence[str] = ["left", "top", "width", "height"]
    ):
        """Create a new instance of BoundingBox from a JSON object
        Args:
            bounding_box (dict): JSON object representing the bounding box
            modifiers (callable): Function to apply to the keys of the JSON object (default: lambda x: x).
            keys (Sequence[str]): Keys of the JSON object (default: ["left", "top", "width", "height"])

        Returns:
            BoundingBox: BoundingBox created from the JSON object
        """
        return BoundingBox(
            left=bounding_box[modifiers(keys[BoundingBoxEnum.LEFT.value])],
            top=bounding_box[modifiers(keys[BoundingBoxEnum.TOP.value])],
            width=bounding_box[modifiers(keys[BoundingBoxEnum.WIDTH.value])],
            height=bounding_box[modifiers(keys[BoundingBoxEnum.HEIGHT.value])],
        )


class Word(BaseModel):
    """Word of a document
    
    Attributes:
        text (str): Text detected in the word
        bounding_boxes (Sequence[BoundingBox]): Bounding boxes of the words in the word
    """
    text: str
    bounding_boxes: BoundingBox
    confidence: Optional[float]

    @validator("confidence")
    def round_confidence(cls, v):
        """Round confidence to 2 decimals"""
        if v is not None:
            return round(v, 2)
        return v

class Line(BaseModel):
    """Line of a document
    
    Attributes:
        text (str): Text detected in the line
        bounding_boxes (Sequence[BoundingBox]): Bounding boxes of the words in the line
    """
    text: str
    words: Sequence[Word] = Field(default_factory=list)
    bounding_boxes: BoundingBox
    confidence: Optional[float]

    @validator("confidence")
    def round_confidence(cls, v):
        """Round confidence to 2 decimals"""
        if v is not None:
            return round(v, 2)
        return v


class Page(BaseModel):
    """Page of a document

    Attributes:
        text (str): Text detected in the page
        bounding_boxes (Sequence[BoundingBox]): Bounding boxes of the words in the page
    """
    lines: Sequence[Line] = Field(default_factory=list)


class OcrAsyncDataClass(BaseModel):
    """OCR Async Data Class
    
    Attributes:
        pages (Sequence[Page]): Pages of the document
    """
    pages: Sequence[Page] = Field(default_factory=list)
    number_of_pages: int

