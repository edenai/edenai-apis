from pydantic import BaseModel, StrictStr
from typing import List, Dict

class FineTuningListProject(BaseModel) :
    listproject : List[Dict]


    