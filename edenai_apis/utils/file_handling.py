import base64
import mimetypes
import random
import tempfile
from typing import AsyncIterator, Iterator

import aiofiles
from curl_cffi.requests import AsyncSession

from edenai_apis.utils.files import FileInfo, FileWrapper
from edenai_apis.utils.http import async_client


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.39 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 OPR/116.0.0.0",
]

BROWSER_IMPERSONATIONS = [
    "chrome",
    "chrome110",
    "chrome120",
    "edge",
    "safari",
]


class FileHandler:

    SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 Mb

    def __init__(self):
        pass

    @staticmethod
    def get_user_agent():
        random_ua = random.choice(USER_AGENTS)
        return {"User-Agent": random_ua}

    @staticmethod
    def get_browser_impersonation() -> str:
        return random.choice(BROWSER_IMPERSONATIONS)

    @staticmethod
    def get_file_extension(mime_type: str) -> str:
        """Derive file extension from MIME type using mimetypes module."""
        clean_mime = mime_type.split(";")[0].strip()
        extension = mimetypes.guess_extension(clean_mime)
        if extension:
            return extension.lstrip(".")
        parts = clean_mime.split("/")
        return parts[-1] if len(parts) > 1 else ""

    @staticmethod
    def parse_content_length(header_value) -> int:
        """Defensively parse Content-Length header, returning -1 on failure."""
        try:
            return int(header_value)
        except (ValueError, TypeError):
            return -1

    @staticmethod
    async def _async_iter_wrapper(sync_iter: Iterator[bytes]) -> AsyncIterator[bytes]:
        """Wrap a sync iterator to make it async-compatible."""
        for chunk in sync_iter:
            yield chunk

    async def _stream_to_file(
        self, chunk_iterator: AsyncIterator[bytes]
    ) -> str:
        """Stream chunks to a temporary file and return the file path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        async with aiofiles.open(tmp_path, "wb") as f:
            async for chunk in chunk_iterator:
                await f.write(chunk)
        return tmp_path

    async def _read_all_chunks(
        self, chunk_iterator: AsyncIterator[bytes]
    ) -> bytes:
        """Read all chunks from iterator into memory."""
        chunks = []
        async for chunk in chunk_iterator:
            chunks.append(chunk)
        return b"".join(chunks)

    async def _build_file_wrapper(
        self,
        file_url: str,
        file_type: str,
        file_size: int,
        chunk_iterator: AsyncIterator[bytes],
        force_file_create: bool,
    ) -> FileWrapper:
        """Common logic for building FileWrapper from chunk iterator."""
        file_wrapper_params = {
            "file_url": file_url,
            "file_info": FileInfo(
                file_size=file_size,
                file_mimetype=file_type,
                file_extension=self.get_file_extension(file_type),
            ),
        }

        if file_size == 0:
            raise Exception("File size is 0")

        if force_file_create or file_size == -1 or file_size > self.SIZE_THRESHOLD:
            tmp_path = await self._stream_to_file(chunk_iterator)
            file_wrapper_params["file_path"] = tmp_path
            file_wrapper_params["file_b64_content"] = None
        else:
            content = await self._read_all_chunks(chunk_iterator)
            b64_content = base64.b64encode(content).decode("utf-8")
            file_wrapper_params["file_path"] = None
            file_wrapper_params["file_b64_content"] = b64_content

        return FileWrapper(**file_wrapper_params)

    async def download_file(
        self, file_url: str, force_file_create: bool = False
    ) -> FileWrapper:
        """
        Download a file from a url and return a FileWrapper.

        Args:
            file_url: URL of the file to download
            force_file_create: if True, always download to disk;
                if False, small files are kept in memory
        """
        try:
            return await self._download_with_httpx(file_url, force_file_create)
        except Exception:
            return await self._download_with_curl_cffi(file_url, force_file_create)

    async def _download_with_httpx(
        self, file_url: str, force_file_create: bool
    ) -> FileWrapper:
        """Primary download method using httpx async_client."""
        async with async_client.stream(
            "GET", file_url, headers=self.get_user_agent()
        ) as response:
            response.raise_for_status()
            return await self._build_file_wrapper(
                file_url=file_url,
                file_type=response.headers.get("Content-Type", "application/octet-stream"),
                file_size=self.parse_content_length(response.headers.get("Content-Length")),
                chunk_iterator=response.aiter_bytes(),
                force_file_create=force_file_create,
            )

    async def _download_with_curl_cffi(
        self, file_url: str, force_file_create: bool
    ) -> FileWrapper:
        """Fallback download method using curl_cffi with browser impersonation."""
        impersonate = self.get_browser_impersonation()

        async with AsyncSession(impersonate=impersonate, timeout=120) as session:
            response = await session.get(file_url, stream=True)
            response.raise_for_status()
            return await self._build_file_wrapper(
                file_url=file_url,
                file_type=response.headers.get("Content-Type", "application/octet-stream"),
                file_size=self.parse_content_length(response.headers.get("Content-Length")),
                chunk_iterator=self._async_iter_wrapper(response.iter_content()),
                force_file_create=force_file_create,
            )
