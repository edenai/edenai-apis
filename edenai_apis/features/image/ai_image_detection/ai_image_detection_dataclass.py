from pydantic import BaseModel, field_validator
from typing import Dict

class C2paAssertion(BaseModel):
    label: str
    data: Dict[str, str]

class C2paSignatureInfo(BaseModel):
    issuer: str
    cert_serial_number: str
    time: str
    timeObject: str

class C2paMetadata(BaseModel):
    claim_generator: str
    title: str
    format: str
    instance_id: str
    ingredients: list[int]
    assertions: list[C2paAssertion]
    signature_info: C2paSignatureInfo
    label: str
    thumbnail: str

class C2paObject(BaseModel):
    activeManifest: C2paMetadata
    manifests: Dict[str, C2paMetadata]

class AiImageDetectionDataClass(BaseModel):
    score: float
    human_probability: float
    ai_probability: float
    version: str
    mime_type: str
    ai_watermark_detected: bool
    c2pa_metadata: C2paObject | None
    iptc_metadata: Dict[str, str] | None

    @staticmethod
    def set_label_based_on_score(ai_score: float):
        if ai_score > 0.5:
            return "ai-generated"
        else:
            return "original"
        
    @field_validator("ai_probability")
    def check_min_max(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Value should be between 0 and 1")
        return v
    
    @staticmethod
    def set_label_based_on_human_score(human_score: float):
        if human_score > 0.5:
            return "original"
        else:
            return "ai-generated"