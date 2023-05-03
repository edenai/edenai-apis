from typing import Dict, Optional
from pydantic import BaseModel, StrictStr

class ChatMessageDataClass(BaseModel):
    user : Optional[StrictStr]
    assistant : Optional[StrictStr]
    
class ChatDataClass(BaseModel):
    generated_text: StrictStr
    message : ChatMessageDataClass = ChatMessageDataClass()

    @staticmethod
    def direct_response(api_response: Dict):
        return api_response["generated_text"]
