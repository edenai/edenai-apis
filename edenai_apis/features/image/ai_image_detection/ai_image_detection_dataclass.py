from pydantic import BaseModel, field_validator
from typing import Dict, Sequence

class ExifIptcItem(BaseModel):
    SpecialInstructions: str
    DateCreated: str
    TimeCreated: str
    Byline: str
    Headline: str
    Credit: str
    Caption: str
    creator: str
    description: Dict[str, str]
    rights: Dict[str, str]
    Instructions: str
    ImageCreator: Dict[str, str]
    Licensor: Dict[str, str]
    UsageTerms: Dict[str, str]
    WebStatement: str
    ImageDescription: str
    XResolution: int
    YResolution: int
    ResolutionUnit: str
    Artist: str
    YCbCrPositioning: int
    Copyright: str

class C2paActions:
    action: str
    digitalSourceType: str
    softwareAgent: str

class C2paData(BaseModel):
    actions: Sequence[Dict[str, str]]

class C2paAssertions(BaseModel):
    label: str
    data: Dict[str, str]

class C2paSignatureInfo(BaseModel):
    issuer: str
    cert_serial_number: str
    time: str

class C2paManifestItem(BaseModel):
    claim_generator: str
    title: str
    format: str
    instance_id: str
    ingredients: Sequence[int]
    assertions: Sequence[C2paAssertions]
    signature_info: C2paSignatureInfo
    label: str

class C2PAItem(BaseModel):
    activeManifest: Dict[str, str]
    manifest: Dict[str, str]

class AiImageDetectionDataClass(BaseModel):
    score: float
    human_probability: float
    ai_probability: float
    version: str
    mime_type: str
    c2pa: C2PAItem | None
    exif: ExifIptcItem | None
    ai_watermark_detected: bool
    ai_watermark_issuers: Sequence[str]


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