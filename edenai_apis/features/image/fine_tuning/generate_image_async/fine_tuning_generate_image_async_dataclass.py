from pydantic import BaseModel
from typing import List

class FineTuningGenerateImageDataClass(BaseModel) :
    images : List[str] #List of all the url of the images