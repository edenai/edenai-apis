from enum import Enum
from http import HTTPStatus
from io import BufferedReader
from json import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional
from warnings import warn

import requests

from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.http import HTTPMethod
from .document import DocumentState, FileParameter, QueryBuilder, UploadDocumentParams
from .models import Document, Organization, Workspace, Collection


class ExtractorType(Enum):
    """This class are used to define the type of extractor

    Actually the affinda api support 6 type of extractor:
        - resume
        - invoice
        - receipt
        - passport
        - credit-note
        - job-description

    Note: Maybe the affinda api support other type of extractor, but these are the only one that are documented.

    Attributes:
        RESUME (str): resume extractor
        INVOICE (str): invoice extractor
        RECEIPT (str): receipt extractor
        PASSPORT (str): passport extractor
        CREDIT_NOTE (str): credit-note extractor
        JOD_DESCRIPTION (str): job-description extractor
    """

    RESUME = "resume"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    PASSPORT = "passport"
    CREDIT_NOTE = "credit-note"
    JOD_DESCRIPTION = "job-description"


class Client:
    """This class are use to simplify the usage of affinda api

    Constants:
        BASE_URL (str): base url for the affinda api (https://api.affinda.com/v3)

    Methods:
        get_organizations: Return all organizations of the current user
        get_organization: Return the organization with the given identifier
        get_workspaces: Return all workspaces of the current_organization property
        get_workspace: Return the workspace with the given identifier
        create_workspace: Create a new workspace
        delete_workspace: Delete the workspace with the given identifier
        get_collections: Return all collections of the current_workspace property
        get_collection: Return the collection with the given identifier
        create_collection: Create a new collection
        delete_collection: Delete the collection with the given identifier
        get_documents: Return all documents of the current_collection property
        get_document: Return the document with the given identifier
        create_document: Create a new document to extract data
        delete_document: Delete the document with the given identifier

    Properties:
        current_organization: The current_organization defines the organization on which the next actions will be performed.
        current_workspace: The current_workspace defines the workspace on which the next actions will be performed.
        current_collection: The current_collection defines the collection on which the next actions will be performed.

    Exemple:
        >>> from edenai_apis.apis.affinda import Client
        >>> client = Client(api_keys="your_api_keys")
        >>> client.current_organization = "your_organization_identifier"
        >>> client.current_workspace = "your_workspace_identifier"
        >>> client.current_collection = "your_collection_identifier"
        >>> response = client.create_document(file="your_file_path")
    """

    BASE_URL: str = "https://api.affinda.com/v3"

    __api_keys: str
    __current_organization: Optional[Organization]
    __current_workspace: Optional[Workspace]
    __current_collection: Optional[Collection]
    __last_api_response: Optional[dict]

    def __init__(self, api_keys: str) -> None:
        self.__api_keys = api_keys
        self.headers = {"Authorization": f"Bearer {self.__api_keys}"}
        self.__current_organization = None
        self.__current_workspace = None
        self.__current_collection = None
        self.__last_api_response = None

    def __requests(
        self,
        method: HTTPMethod,
        url: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        files: Optional[dict] = None,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        """This function are use to simplify the usage of requests module with the affinda api error management

        Args:
            method (HTTPMethod): method for the new `Request` object: `GET`, `POST`, `PUT`, `PATCH`, or `DELETE`.
            url (str): URL for the new `Request` object.
            params (dict, optional): Dictionary to send in the query string for the `Request`.
            data (dict, optional): Dictionary to send in the body of the `Request`.
            json (dict, optional): A JSON serializable Python object to send in the body of the `Request`.
            headers (dict, optional): Dictionary of HTTP Headers to send with the `Request`.
            files (optional): Dictionary of `'name': file-like-objects` (or `{'name': file-tuple}`) for multipart encoding upload.
            `file-tuple` can be a 2-tuple `('filename', fileobj)`, 3-tuple `('filename', fileobj, 'content_type')`
            or a 4-tuple `('filename', fileobj, 'content_type', custom_headers)`, where `'content-type'` is a string
            defining the content type of the given file and `custom_headers` a dict-like object containing additional headers
            to add for the file.

        Raises:
            ProviderException: if HTTPError or JSONDecodeError are raised
            RequestException: if an error are raised by requests module

        Returns:
            dict: The response of the request in json format. If status_code is 204, return { 'status_code': 204 }
        """
        response: requests.Response = requests.request(
            method=method.value,
            url=url,
            data=data,
            params=params,
            files=files,
            headers=headers,
            json=json,
        )

        try:
            response.raise_for_status()

            if response.status_code == HTTPStatus.NO_CONTENT:
                return {"status_code": response.status_code}

            self.__last_api_response = response.json()
            return self.__last_api_response
        except requests.exceptions.HTTPError as exc:
            raise ProviderException(
                message=f"{exc}\nError message: {exc.response.text}",
                code=response.status_code,
            ) from exc
        except JSONDecodeError:
            raise ProviderException(
                message="Internal server error", code=response.status_code
            )

    @property
    def last_api_response(self):
        """The last_api_response property."""
        return self.__last_api_response

    def get_organizations(self) -> List[Organization]:
        """Return all organizations of the current user"""
        return [
            Organization(**organization)
            for organization in self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/organizations",
                headers=self.headers,
            )
        ]

    def get_organization(self, identifier: str) -> Organization:
        """Return the organization with the given identifier

        Args:
            identifier (str): The identifier of the organization

        Return:
            Organization: The organization with the given identifier
        """
        return Organization(
            **self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/organizations/{identifier}",
                headers=self.headers,
            )
        )

    @property
    def current_organization(self) -> Optional[Organization]:
        """The current_organization defines the organization on which the next actions will be performed.

        Get:
            Return the current_organization property

        Set:
            Set the current_organization property by the given identifier

        Exemples:
            >>> from edenai_apis.apis.affinda import client
            >>> client.current_organization = "your_organization_identifier"
            >>> print(client.current_organization)
            <<< Organization(identifier="your_organization_identifier", name="your_organization_name")
        """
        return self.__current_organization

    @current_organization.setter
    def current_organization(self, identifier: str) -> None:
        self.__current_organization = self.get_organization(identifier)

    @current_organization.deleter
    def current_organization(self) -> None:
        self.__current_organization = None

    def get_workspaces(self) -> List[Workspace]:
        """Return all workspaces of the current_organization property."""
        if not self.__current_organization:
            warn(
                "You didn't have set a current_organization. Please see the documentations."
            )
            return []
        query = {"organization": self.__current_organization.identifier}
        return [
            Workspace(**workspace)
            for workspace in self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/workspaces",
                headers=self.headers,
                params=query,
            )
        ]

    def get_workspace(self, identifier: str) -> Workspace:
        """Return the workspace with the given identifier.

        Args:
            identifier (str): The identifier of the workspace.

        Returns:
            Workspace: The workspace with the given identifier.
        """
        return Workspace(
            **self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/workspaces/{identifier}",
                headers=self.headers,
            )
        )

    @property
    def current_workspace(self) -> Optional[Workspace]:
        """The current_workspace defines the workspace on which the next actions will be performed.

        Get:
            Return the current_workspace property

        Set:
            Set the current_workspace property by the given identifier

        Exemples:
            >>> from edenai_apis.apis.affinda import client
            >>> client.current_organization = "your_organization_identifier"
            >>> client.current_workspace = "your_workspace_identifier"
            >>> print(client.current_workspace)
            <<< Workspace(identifier="your_workspace_identifier", name="your_workspace_name")
        """
        return self.__current_workspace

    @current_workspace.setter
    def current_workspace(self, identifier: str) -> None:
        self.__current_workspace = self.get_workspace(identifier)

    @current_workspace.deleter
    def current_workspace(self) -> None:
        self.__current_workspace = None

    def create_workspace(
        self, name: str, visibility: Literal["organization", "private"] = "organization"
    ) -> Workspace:
        """Create a new workspace in the current_organization property.

        Args:
            name (str): The name of the workspace.
            visibility (Literal["organization", "private"], optional): The visibility of the workspace. Defaults to "organization".

        Returns:
            Workspace: The new workspace.
        """
        if not self.__current_organization:
            raise AttributeError(
                "You didn't have set a current_organization. Please see the documentations."
            )
        payload = {
            "name": name,
            "visibility": visibility,
            "organization": self.__current_organization.identifier,
        }
        return Workspace(
            **self.__requests(
                method=HTTPMethod.POST,
                url=f"{self.BASE_URL}/workspaces",
                headers=self.headers,
                json=payload,
            )
        )

    def delete_workspace(self, identifier: str) -> None:
        """Delete the workspace with the given identifier.

        Args:
            identifier (str): The identifier of the workspace.
        """
        self.__requests(
            method=HTTPMethod.DELETE,
            url=f"{self.BASE_URL}/workspaces/{identifier}",
            headers=self.headers,
        )

    def get_collections(self) -> List[Collection]:
        """Return all collections of the current_workspace property."""
        if not self.__current_workspace:
            warn(
                "You didn't have set a current_workspace. Please see the documentations."
            )
            return []
        query = {"workspace": self.__current_workspace.identifier}
        return [
            Collection(**collection)
            for collection in self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/collections",
                headers=self.headers,
                params=query,
            )
        ]

    def get_collection(self, identifier: str) -> Collection:
        """Return the collection with the given identifier.

        Args:
            identifier (str): The identifier of the collection.
        """
        return Collection(
            **self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/collections/{identifier}",
                headers=self.headers,
            )
        )

    @property
    def current_collection(self) -> Optional[Collection]:
        """The current_collection defines the collection on which the next actions will be performed.

        If new collection was created, the current_workspace are updated.

        Get:
            Return the current_collection property

        Set:
            Set the current_collection property by the given identifier

        Exemples:
            >>> from edenai_apis.apis.affinda import client
            >>> client.current_organization = "your_organization_identifier"
            >>> client.current_workspace = "your_workspace_identifier"
            >>> client.current_collection = "your_collection_identifier"
            >>> print(client.current_collection)
            <<< Collection(identifier="your_collection_identifier", name="your_collection_name")
        """
        return self.__current_collection

    @current_collection.setter
    def current_collection(self, identifier: str) -> None:
        self.__current_collection = self.get_collection(identifier)

    @current_collection.deleter
    def current_collection(self) -> None:
        self.__current_collection = None

    def create_collection(self, name: str, extractor: ExtractorType) -> Collection:
        """Create a new collection in the current_workspace property.

        Args:
            name (str): The name of the collection.
            extractor (ExtractorType): The extractor of the collection

        Returns:
            Collection: The new collection.
        """
        if not self.__current_workspace:
            raise AttributeError(
                "You didn't have set a current_workspace. Please see the documentations."
            )

        payload = {
            "name": name,
            "workspace": self.__current_workspace.identifier,
            "extractor": extractor.value,
        }
        new_collection: Collection = Collection(
            **self.__requests(
                method=HTTPMethod.POST,
                url=f"{self.BASE_URL}/collections",
                headers=self.headers,
                json=payload,
            )
        )
        self.current_collection = new_collection.identifier
        return new_collection

    def delete_collection(self, identifier: str) -> None:
        """Delete the collection with the given identifier.
        Args:
            identifier (str): The identifier of the collection.
        """
        self.__requests(
            method=HTTPMethod.DELETE,
            url=f"{self.BASE_URL}/collections/{identifier}",
            headers=self.headers,
        )
        if self.__current_workspace:
            self.current_workspace = self.__current_workspace.identifier

    def get_documents(self, state: Optional[DocumentState] = None) -> List[dict]:
        """Return all documents of the current_collection property.

        Args:
            state (Optional[DocumentState], optional): The state of the documents. Defaults to None.
        """
        query = (
            QueryBuilder()
            .add_state(state)
            .add_workspace(self.__current_workspace)
            .add_collection(self.__current_collection)
            .build()
        )

        return [
            result
            for result in self.__requests(
                method=HTTPMethod.GET,
                url=f"{self.BASE_URL}/documents",
                headers=self.headers,
                params=query,
            )["results"]
        ]

    def get_document(self, identifier: str) -> dict:
        """Return the document with the given identifier.
        Args:
            identifier (str): The identifier of the document.
        """
        return self.__requests(
            method=HTTPMethod.GET,
            url=f"{self.BASE_URL}/documents/{identifier}",
            headers=self.headers,
        )

    def create_document(
        self,
        file: FileParameter,
        parameters: UploadDocumentParams = UploadDocumentParams(),
    ) -> Document:
        """Create a new document in the current_collection property.
        Args:
            file (FileParameter): The file to upload.
            parameters (UploadDocumentParams, optional): The parameters of the document. Defaults to UploadDocumentParams().
        Returns:
            dict: The new document.
        """
        payload: Dict[str, Any] = {
            **parameters.to_form_data(),
            **(
                QueryBuilder()
                .add_workspace(self.__current_workspace)
                .add_collection(self.__current_collection)
                .build()
            ),
        }

        files: Optional[Dict[str, BufferedReader]] = None

        if file.type == "file":
            with open(file.file, "rb") as f:
                files = {"file": f}
                return Document(
                    **self.__requests(
                        method=HTTPMethod.POST,
                        url=f"{self.BASE_URL}/documents",
                        headers=self.headers,
                        files=files,
                        data=payload,
                    )
                )
        else:
            payload["url"] = file.file
            return Document(
                **self.__requests(
                    method=HTTPMethod.POST,
                    url=f"{self.BASE_URL}/documents",
                    headers=self.headers,
                    files=None,
                    data=payload,
                )
            )

    def delete_document(self, identifier: str) -> None:
        """Delete the document with the given identifier.

        Args:
            identifier (str): The identifier of the document.
        """
        self.__requests(
            method=HTTPMethod.DELETE,
            url=f"{self.BASE_URL}/documents/{identifier}",
            headers=self.headers,
        )
