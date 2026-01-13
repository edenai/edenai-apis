import mimetypes
import os
from enum import Enum
from json import JSONDecodeError
from typing import Optional, Dict

import httpx
import aiofiles

from edenai_apis.utils.exception import ProviderException
from .models import ResponseData


class Parser(Enum):
    RESUME = "resume"
    JD = "job_description"


class AsyncClient:
    """
    An asynchronous client for the Senseloaf API.

    Use the `AsyncClient.create()` factory method to instantiate the client,
    as it needs to perform an async login to retrieve the API key.
    """

    BASE_URL = "https://service.senseloaf.com"

    def __init__(self, api_key: str, client: httpx.AsyncClient):
        self._api_key = api_key
        self._client = client
        self._last_api_response: Optional[Dict] = None
        self._last_api_response_type: Optional[str] = None
        self._last_api_response_code: Optional[str] = None

    @classmethod
    async def create(
        cls,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> "AsyncClient":
        client = httpx.AsyncClient(timeout=30.0)
        api_key = await cls._login_async(email, password, client)
        return cls(api_key, client)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Closes the underlying httpx client session."""
        await self._client.aclose()

    @staticmethod
    async def _login_async(email: str, password: str, client: httpx.AsyncClient) -> str:
        """Performs the login request to get an API token."""
        url = f"{AsyncClient.BASE_URL}/login"
        headers = {"Content-Type": "application/json"}
        payload = {"emailId": email, "password": password}

        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        auth_header = response.headers.get("Authorization")
        if not auth_header:
            raise ProviderException(
                "Login failed: Authorization token not found in response."
            )

        return auth_header.replace("Bearer ", "")

    async def _request_async(
        self,
        method: str,
        url: str,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        json_field: Optional[dict] = None,
        files: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> httpx.Response:
        """Internal method to handle all HTTP requests."""
        try:
            response = await self._client.request(
                method=method,
                url=url,
                data=data,
                params=params,
                files=files,
                headers=headers,
                json=json_field,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            try:
                message = exc.response.json()
                if message.get("errorCode") == "AUTHENTICATION_FAILED":
                    message["errorMessage"] = (
                        "Authentication Failed. Please check your credentials or API Key."
                    )
                raise ProviderException(
                    message=f"{exc}\nError message: {message}",
                    code=str(exc.response.status_code),
                ) from exc
            except JSONDecodeError:
                raise ProviderException(
                    message="Internal server error", code=str(exc.response.status_code)
                ) from exc

    async def _parse_and_handle_response(
        self, response: httpx.Response, return_type: str = "json"
    ) -> ResponseData:
        """Handles parsing the httpx.Response into a ResponseData object."""
        response_data = {}
        if return_type == "json":
            response_data = response.json()
        # Add other return types if needed (e.g., 'content', 'text')

        self._last_api_response = response_data
        self._last_api_response_code = str(response.status_code)
        self._last_api_response_type = return_type

        return ResponseData(
            response=response_data,
            response_code=str(response.status_code),
            response_type=return_type,
        )

    async def _parse_resume_from_file_async(
        self,
        file: str,
        file_content: Optional[bytes] = None,
        mime_type: Optional[str] = None,
    ) -> ResponseData:
        url = f"{self.BASE_URL}/api/v2/parse-resume"

        if file_content is None:
            async with aiofiles.open(file, "rb") as f:
                file_content = await f.read()

        filename = os.path.basename(file)
        if mime_type is None:
            mime_type = mimetypes.guess_type(file)[0] or "application/octet-stream"

        files = {"files": (filename, file_content, mime_type)}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        response = await self._request_async("POST", url, headers=headers, files=files)
        return await self._parse_and_handle_response(response)

    async def _parse_jd_from_file_async(self, file: str) -> ResponseData:
        url = f"{self.BASE_URL}/api/parse-jd"
        async with aiofiles.open(file, "rb") as f:
            file_content = await f.read()

        filename = os.path.basename(file)
        mime_type = mimetypes.guess_type(file)[0] or "application/octet-stream"

        files = {"files": (filename, file_content, mime_type)}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        response = await self._request_async("POST", url, headers=headers, files=files)
        return await self._parse_and_handle_response(response)

    async def parse_document_async(
        self,
        parse_type: Parser,
        file: Optional[str] = "",
        file_content: Optional[bytes] = None,
        mime_type: Optional[str] = None,
    ) -> ResponseData:
        if not file and not file_content:
            raise ProviderException("Please provide either a file path or file content.")

        if parse_type.value == "resume":
            return await self._parse_resume_from_file_async(file, file_content, mime_type)

        if parse_type == Parser.JD:
            if file:
                return await self._parse_jd_from_file_async(file)
            raise ProviderException(
                "Please provide a file path for Job Description parsing."
            )

        # Handle other parser types
        else:
            raise NotImplementedError(
                f"Parsing for '{parse_type.value}' is not implemented yet. "
                "Reach out to us at team@senseloaf.com for requesting early release."
            )

    @property
    def cache(self):
        return ResponseData(
            response=self._last_api_response,
            response_code=self._last_api_response_code,
            response_type=self._last_api_response_type,
        )

    def clear_cache(self):
        self._last_api_response = None
        self._last_api_response_code = None
        self._last_api_response_type = None
