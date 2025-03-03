from enum import IntEnum
from typing import Callable, Dict, List, Sequence

from pydantic import BaseModel, Field, field_validator
from typing_extensions import overload


class BoundingBoxEnum(IntEnum):
    """Enum for the keys of the bounding box JSON object

    Attributes:
        LEFT (int): Index of the left coordinate of the bounding box
        TOP (int): Index of the top coordinate of the bounding box
        WIDTH (int): Index of the width of the bounding box
        HEIGHT (int): Index of the height of the bounding box
    """

    LEFT = 0
    TOP = 1
    WIDTH = 2
    HEIGHT = 3


class BoundingBoxCornerEnum(IntEnum):
    """Enum for the corners of the bounding box

    Attributes:
        TOP_LEFT (int): Index of the top left corner of the bounding box
        TOP_RIGHT (int): Index of the top right corner of the bounding box
        BOTTOM_LEFT (int): Index of the bottom left corner of the bounding box
        BOTTOM_RIGHT (int): Index of the bottom right corner of the bounding box
    """

    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


class CoordinateEnum(IntEnum):
    """Enum for the keys of the coordinate JSON object

    Attributes:
        X (int): Index of the X coordinate of the bounding box
        Y (int): Index of the Y coordinate of the bounding box
    """

    X = 0
    Y = 1


class BoundingBox(BaseModel):
    """Bounding box of a word in the image

    Attributes:
        left (float): Left coordinate of the bounding box
        top (float): Top coordinate of the bounding box
        width (float): Width of the bounding box
        height (float): Height of the bounding box
        text (str): Text detected in the bounding box

    Constructor:
        from_json (classmethod): Create a new instance of BoundingBox from a JSON object
        from_normalized_vertices (classmethod): Create a new instance of BoundingBox from normalized vertices
        unknown (classmethod): Return a invalid bouding_box with all field filled with `-1`
    """

    left: float = Field(description="Left coordinate of the bounding box")
    top: float = Field(description="Top coordinate of the bounding box")
    width: float = Field(description="Width of the bounding box")
    height: float = Field(description="Height of the bounding box")

    @field_validator("left", "top", "width", "height", mode="before")
    def convert_to_float(cls, value):
        """Convert the value to float

        Args:
            value (Any): Value to convert

        Returns:
            float: Converted value
        """
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Value {value} cannot be converted to float") from exc

    @classmethod
    def unknown(cls) -> "BoundingBox":
        return cls(left=-1, top=-1, width=-1, height=-1)

    @classmethod
    def from_json(
        cls,
        bounding_box: dict,
        modifiers: Callable = lambda x: x,
        keys: Sequence[str] = ["left", "top", "width", "height"],
    ) -> "BoundingBox":
        """Create a new instance of BoundingBox from a JSON object
        Args:
            bounding_box (dict): JSON object representing the bounding box
            modifiers (callable): Function to apply to the keys of the JSON object (default: lambda x: x).
            keys (Sequence[str]): Keys of the JSON object (default: ["left", "top", "width", "height"])

        Returns:
            BoundingBox: BoundingBox created from the JSON object
        """
        return cls(
            left=bounding_box[modifiers(keys[BoundingBoxEnum.LEFT.value])],
            top=bounding_box[modifiers(keys[BoundingBoxEnum.TOP.value])],
            width=bounding_box[modifiers(keys[BoundingBoxEnum.WIDTH.value])],
            height=bounding_box[modifiers(keys[BoundingBoxEnum.HEIGHT.value])],
        )

    @classmethod
    @overload
    def from_normalized_vertices(
        cls,
        normalized_vertices: Dict[str, float],
        coordinate_keys: Sequence[str] = ["x", "y"],
        corner_position_keys: Sequence[str] = [
            "topLeft",
            "topRight",
            "bottomLeft",
            "bottomRight",
        ],
    ) -> "BoundingBox": ...

    @classmethod
    @overload
    def from_normalized_vertices(
        cls,
        normalized_vertices: List[float],
        coordinate_keys: Sequence[str] = ["x", "y"],
        corner_position_keys: Sequence[int] = [0, 1, 2, 3],
    ) -> "BoundingBox": ...

    @classmethod
    def from_normalized_vertices(
        cls,
        normalized_vertices,
        coordinate_keys=["x", "y"],
        corner_position_keys=[],
    ) -> "BoundingBox":
        """Create a new instance of BoundingBox from a list of normalized vertices

        Args:
            normalized_vertices (list): List of normalized vertices
            coordinate_keys (Sequence[str | int]): Keys of the JSON object (default: ["x", "y"]).
            corner_position_keys (Sequence[int | str]): Keys of the JSON object.

        Returns:
            BoundingBox: BoundingBox created from the list of normalized vertices.

        Dependencies:
            BoundingBox.from_json
        """
        if corner_position_keys == []:
            if isinstance(normalized_vertices, dict):
                corner_position_keys = [
                    "topLeft",
                    "topRight",
                    "bottomLeft",
                    "bottomRight",
                ]
            elif isinstance(normalized_vertices, list):
                corner_position_keys = [0, 1, 2, 3]
            else:
                raise ValueError("normalized_vertices must be a dict or string")

        boxes = {}

        x_key = coordinate_keys[CoordinateEnum.X.value]
        y_key = coordinate_keys[CoordinateEnum.Y.value]

        top_left_key = corner_position_keys[BoundingBoxCornerEnum.TOP_LEFT.value]
        top_right_key = corner_position_keys[BoundingBoxCornerEnum.TOP_RIGHT.value]
        bottom_left_key = corner_position_keys[BoundingBoxCornerEnum.BOTTOM_LEFT.value]

        xleft = normalized_vertices[top_left_key][x_key]
        xright = normalized_vertices[top_right_key][x_key]
        ytop = normalized_vertices[top_left_key][y_key]
        ybottom = normalized_vertices[bottom_left_key][y_key]

        boxes = {
            "left": xleft,
            "top": ytop,
            "width": xright - xleft,
            "height": ybottom - ytop,
        }

        return cls.from_json(boxes)
