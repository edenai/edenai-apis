"""Module for Affinda API models.

These models are used as DTOs for affinda api responses.

NOTE: Please note that these models do not represent the complete API responses, only what can be used.
      If any fields are missing, don't hesitate to add them.
"""
from typing import Literal, Optional, Sequence
from pydantic import BaseModel, Field


class Organization(BaseModel):
    identifier: str
    name: str
    avatar: Optional[str]
    is_trial: bool = Field(..., alias="isTrial")
    resthook_signature_key: Optional[str] = Field(..., alias="resthookSignatureKey")

class Extractor(BaseModel):
    identifier: str
    name: str
    category: str
    validatable: bool
    is_custom: Optional[bool] = Field(alias="isCustom", default=None)
    has_custom_data_points: bool = Field(..., alias="hasCustomDataPoints")

class Collection(BaseModel):
    identifier: str
    name: str
    extractor: Extractor

class Workspace(BaseModel):
    identifier: str
    organization: Organization
    collections: Sequence[Collection]
    name: str
    visibility: Literal["organization", "private"]
    reject_invalid_documents: bool = Field(..., alias="rejectInvalidDocuments")
    reject_duplicates: bool = Field(..., alias="rejectDuplicates")
    split_documents: bool = Field(..., alias="splitDocuments")
