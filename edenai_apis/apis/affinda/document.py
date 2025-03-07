# TODO: Add tests for this file
from enum import Enum
from typing import Any, Dict, Optional, Literal

from .models import Workspace, Collection


class DocumentState(Enum):
    """State of the document

    Attributes:
        UPLOADED: Document has been uploaded
        REVIEW: Document is being reviewed
        VALIDATED: Document has been validated
        ARCHIVED: Document has been archived
        REJECTED: Document has been rejected
    """

    UPLOADED = "uploaded"
    REVIEW = "review"
    VALIDATED = "validated"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class QueryBuilder:
    """Builder for the query parameters

    Attributes:
        query (dict): Query parameters

    Methods:
        reset: Reset the query parameters
        add_state: Add the state parameter to the query
        add_workspace: Add the workspace parameter to the query
        add_collection: Add the collection parameter to the query
        build: Build the query parameters
    """

    __query: Dict[str, Any]

    def __init__(self) -> None:
        self.__query = {}

    def reset(self) -> "QueryBuilder":
        """Reset the query parameters"""
        self.__query = {}
        return self

    def add_state(self, state: Optional[DocumentState], required: bool = False):
        """Add the state parameter to the query.
        If required is set to True, the state parameter will be added even if it is None

        Args:
            state (DocumentState): State of the document
            required (bool): Whether the state parameter is required
        """
        if required is True or state is not None:
            self.__query["state"] = state.value if state else None
        return self

    def add_workspace(self, workspace: Optional[Workspace], required: bool = False):
        """Add the workspace parameter to the query.
        If required is set to True, the workspace parameter will be added even if it is None

        Args:
            workspace (Workspace): Workspace of the document
            required (bool): Whether the workspace parameter is required
        """
        if required is True or workspace is not None:
            self.__query["workspace"] = workspace.identifier if workspace else None
        return self

    def add_collection(self, collection: Optional[Collection], required: bool = False):
        """Add the collection parameter to the query.
        If required is set to True, the collection parameter will be added even if it is None

        Args:
            collection (Collection): Collection of the document
            required (bool): Whether the collection parameter is required
        """
        if required is True or collection is not None:
            self.__query["collection"] = collection.identifier if collection else None
        return self

    def build(self) -> Dict[str, Any]:
        """Build the query parameters

        Returns:
            dict: Query parameters ready to be passed to the request
        """
        return self.__query


class FileParameter:
    """File parameter for document upload

    Attributes:
        file (str): Path to the file to upload
        type (str): Type of the file, either 'file' or 'url'
    """

    __file: str
    __type: Literal["file", "url"]

    def __init__(self, file: Optional[str], url: Optional[str]) -> None:
        if url:
            self.__file = url
            self.__type = "url"
        elif file:
            self.__file = file
            self.__type = "file"
        else:
            raise ValueError("File or URL must be provided")

    @property
    def file(self) -> str:
        """Path or Url to the file to upload"""
        return self.__file

    @property
    def type(self) -> Literal["file", "url"]:
        """Type of the file, either 'file' or 'url'"""
        return self.__type


class UploadDocumentParams:
    """Parameters for document upload

    Attributes:
        wait (bool): Wait for the document to be processed before returning
        identifier (str): Custom Identifier of the document
        filename (str): Filename of the document
        expiry_time (str): Expiry time of the document
        language (str): Language of the document
        reject_duplicates (bool): Reject duplicates of the document

    Methods:
        to_form_data: Convert the parameters to a dictionary to be passed to the request
    """

    __wait: bool
    __identifier: Optional[str]
    __filename: Optional[str]
    __expiry_time: Optional[str]
    __language: Optional[str]
    __reject_duplicates: Optional[bool]

    def __init__(
        self,
        wait: bool = True,
        identifier: Optional[str] = None,
        filename: Optional[str] = None,
        expiry_time: Optional[str] = None,
        language: Optional[str] = None,
        reject_duplicates: Optional[bool] = None,
    ) -> None:
        self.__wait = wait
        self.__identifier = identifier
        self.__filename = filename
        self.__expiry_time = expiry_time
        self.__language = language
        self.__reject_duplicates = reject_duplicates

    @property
    def wait(self) -> dict:
        """Wait for the document to be processed before returning"""
        return {"wait": self.__wait}

    @property
    def identifier(self) -> dict:
        """Custom Identifier of the document"""
        if self.__identifier is None:
            return {}
        return {"identifier": self.__identifier}

    @property
    def filename(self) -> dict:
        """Filename of the document"""
        if self.__filename is None:
            return {}
        return {"filename": self.__filename}

    @property
    def expiry_time(self) -> dict:
        """Expiry time of the document"""
        if self.__expiry_time is None:
            return {}
        return {"expiryTime": self.__expiry_time}

    @property
    def language(self) -> dict:
        """Language of the document"""
        if self.__language is None:
            return {}
        return {"language": self.__language}

    @property
    def reject_duplicates(self) -> dict:
        """Reject duplicates of the document"""
        if self.__reject_duplicates is None:
            return {}
        return {"rejectDuplicates": self.__reject_duplicates}

    def to_form_data(self) -> Dict[str, Any]:
        """Convert the parameters to a dictionary to be passed to the request"""
        return {
            **self.wait,
            **self.identifier,
            **self.filename,
            **self.expiry_time,
            **self.language,
            **self.reject_duplicates,
        }
