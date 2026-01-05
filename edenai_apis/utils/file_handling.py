import base64
import random
import tempfile

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

    async def download_file(self, file_url: str, force_file_create: bool = False) -> FileWrapper:
        """
        Download a file from a url and return the file path

        Args:
            file_url (str): url of the file to download
            force_file_create: if True, the file will be downloaded to disk;
                if False, the file will be downloaded to memory
        """
        try:
            return await self._download_with_httpx(file_url, force_file_create)
        except Exception:
            # Fallback to curl_cffi with browser impersonation to bypass TLS fingerprinting
            return await self._download_with_curl_cffi(file_url, force_file_create)

    async def _download_with_httpx(self, file_url: str, force_file_create: bool) -> FileWrapper:
        """Primary download method using httpx async_client."""
        response = await async_client.head(
            file_url, headers=FileHandler.get_user_agent()
        )
        response.raise_for_status()
        file_type = response.headers.get("Content-Type", "application/octet-stream")
        file_size = int(response.headers.get("Content-Length", -1))

        file_wrapper_params = {
            "file_url": file_url,
            "file_info": FileInfo(
                file_size=file_size,
                file_mimetype=file_type,
                file_extension=file_type.split("/")[-1],
            ),
        }

        if file_size == 0:
            raise Exception("File size is 0")

        if force_file_create or file_size == -1 or file_size > self.SIZE_THRESHOLD:
            async with async_client.stream(
                "GET", file_url, headers=FileHandler.get_user_agent()
            ) as stream:
                stream.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = tmp.name
                async with aiofiles.open(tmp_path, "wb") as f:
                    async for chunk in stream.aiter_bytes():
                        await f.write(chunk)
                file_wrapper_params["file_path"] = tmp_path
            file_wrapper_params["file_b64_content"] = None
            return FileWrapper(**file_wrapper_params)

        response = await async_client.get(
            file_url, headers=FileHandler.get_user_agent()
        )
        response.raise_for_status()

        b64_content = base64.b64encode(response.content).decode("utf-8")
        file_wrapper_params["file_path"] = None
        file_wrapper_params["file_b64_content"] = b64_content
        return FileWrapper(**file_wrapper_params)

    async def _download_with_curl_cffi(self, file_url: str, force_file_create: bool) -> FileWrapper:
        """Fallback download method using curl_cffi with browser impersonation."""
        impersonate = self.get_browser_impersonation()

        async with AsyncSession(impersonate=impersonate, timeout=120) as session:
            response = await session.head(file_url)
            response.raise_for_status()
            file_type = response.headers.get("Content-Type", "application/octet-stream")
            file_size = int(response.headers.get("Content-Length", -1))

            file_wrapper_params = {
                "file_url": file_url,
                "file_info": FileInfo(
                    file_size=file_size,
                    file_mimetype=file_type,
                    file_extension=file_type.split("/")[-1],
                ),
            }

            if file_size == 0:
                raise Exception("File size is 0")

            if force_file_create or file_size == -1 or file_size > self.SIZE_THRESHOLD:
                response = await session.get(file_url, stream=True)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = tmp.name
                async with aiofiles.open(tmp_path, "wb") as f:
                    for chunk in response.iter_content():
                        await f.write(chunk)
                file_wrapper_params["file_path"] = tmp_path

                file_wrapper_params["file_b64_content"] = None
                return FileWrapper(**file_wrapper_params)

            response = await session.get(file_url)
            response.raise_for_status()

            b64_content = base64.b64encode(response.content).decode("utf-8")
            file_wrapper_params["file_path"] = None
            file_wrapper_params["file_b64_content"] = b64_content
            return FileWrapper(**file_wrapper_params)
