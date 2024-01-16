
from pydantic import BaseModel, StrictStr


class FineTuningCreateProjectDataClass(BaseModel) :
    projectid : int
    name : StrictStr
    description : StrictStr