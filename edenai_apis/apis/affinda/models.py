"""Module for Affinda API models.

These models are used as DTOs for affinda api responses.

NOTE: Please note that these models do not represent the complete API responses, only what can be used.
      If any fields are missing, don't hesitate to add them.
"""

from typing import Any, Dict, Literal, Optional, Sequence

from pydantic import BaseModel, Field


class Organization(BaseModel):
    """This model are used to represent an Organization

    Fields:
        identifier (str): The organization's unique id
        name (str): The organization's unique name
        avatar (str, optional): The url of the organization's avatar
        is_trial (bool): A boolean to describe if the organization have a trial account or not (alias -> isTrial)
        resthook_signature_key (str, optional): The organization's signature
    """

    identifier: str
    name: str
    avatar: Optional[str]
    is_trial: bool = Field(..., alias="isTrial")
    resthook_signature_key: Optional[str] = Field(..., alias="resthookSignatureKey")


class Extractor(BaseModel):
    """This model are used to represent

    Fields:
        identifier (str): The extractor's unique id
        name (str): The extractor's unique name
        validatable (bool): A bool to describe if the extractor can be validated or not
        category (str, optional): The extractor's category (Ex: 'resume', 'invoice')
        is_custom (bool, optional): A bool to describe if the extractor a custom one (alias -> isCustom)
        has_custom_data_points (bool): A bool to describe if the extractor has custom data points (alias -> hasCustomDataPoints)
    """

    identifier: str
    name: str
    validatable: bool
    category: Optional[str] = Field(default=None)
    is_custom: Optional[bool] = Field(alias="isCustom", default=None)
    has_custom_data_points: bool = Field(..., alias="hasCustomDataPoints")


class Collection(BaseModel):
    """This model are used to represent a collection

    Fields:
       identifier (str): The collection's unique id
       name (str): The collection's unique name
       extractor (Extractor): The extractor associated at the collection. See the Extractor's documentation
    """

    identifier: str
    name: str
    extractor: Extractor


class Workspace(BaseModel):
    """This model are used to represent a workspace

    Fields:
        identifier (str): The collection's unique id
        name (str): The collection's unique name
        organization (Or)
    """

    identifier: str
    name: str
    organization: Organization
    collections: Sequence[Collection]
    visibility: Literal["organization", "private"]
    reject_invalid_documents: bool = Field(..., alias="rejectInvalidDocuments")
    reject_duplicates: bool = Field(..., alias="rejectDuplicates")
    split_documents: bool = False


class DocumentError(BaseModel):
    """This model are used to represent the error during the upload or the parsing of a document.
    If no error was occurred, code and detail are set at None

    Fields:
        code (int, optional): The error's status code
        detail (str, optional): The error's detail
    """

    code: Optional[str] = Field(..., alias="errorCode")
    detail: Optional[str] = Field(..., alias="errorDetail")


class DocumentMeta(BaseModel):
    """This model are used to represent all the document's metadata like filename or id

    Fields:
        identifier (str): The document's unique id
        filename (str): The document's filename
        is_archived (bool): A boolean to describe if the document are archived
        is_confirmed (bool): A boolean to describe if the document are confirmed
        is_ocrd (bool): A boolean to describe if the document are analysed
        is_rejected (bool): A boolean to describe if the document are rejected
        is_failed (bool): A boolean to describe if the document's parsing are failed
        is_ready (bool): A boolean to describe if the document's parsing are done
        ready_date (str): Date at the document's parsing was finished
        review_url (url): Url of the document's review
        expiry_time (str, optional): The date from which the file will expire
    """

    identifier: str
    filename: str = Field(..., alias="fileName")
    is_archived: bool = Field(..., alias="isArchived")
    is_confirmed: bool = Field(..., alias="isConfirmed")
    is_ocrd: bool = Field(..., alias="isOcrd")
    is_rejected: bool = Field(..., alias="isRejected")
    is_failed: bool = Field(..., alias="failed")
    is_ready: bool = Field(..., alias="ready")
    ready_date: str = Field(..., alias="readyDt")
    review_url: str = Field(..., alias="reviewUrl")
    expiry_time: Optional[str] = Field(..., alias="expiryTime")


class Document(BaseModel):
    """This model are used to represent a Document

    Fields:
        data (dict): The result of the document analysis
        extractor (str): The extractor's category was use for the parsing
        error (DocumentError): An instance of DocumentError to describe a potential error
        meta (DocumentMeta): An instance of DocumentMeta to describe all of meta data
    """

    data: Dict[str, Any]
    extractor: str
    error: DocumentError
    meta: DocumentMeta
