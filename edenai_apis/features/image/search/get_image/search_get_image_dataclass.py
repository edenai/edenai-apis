from io import BytesIO
from sre_constants import ANY
from typing import Any, ByteString
from pydantic import BaseModel


class SearchGetImageDataClass(BaseModel):
    image: bytes
