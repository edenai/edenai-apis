import base64
import mimetypes
import random
import tempfile

from curl_cffi.requests import AsyncSession

from edenai_apis.utils.files import FileInfo, FileWrapper


# Browser fingerprints to impersonate (bypasses TLS fingerprinting)
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
    def get_browser_impersonation() -> str:
        """Get a random browser fingerprint to impersonate."""
        return random.choice(BROWSER_IMPERSONATIONS)

    async def download_file(
        self, file_url: str, force_file_create: bool = False
    ) -> FileWrapper:
        """
        Download a file from a url and return the file path.

        Uses curl_cffi with browser impersonation to bypass TLS fingerprinting
        that blocks standard Python HTTP clients (httpx, aiohttp, requests).

        Args:
            file_url (str): url of the file to download
            force_file_create: if True, the file will be downloaded to disk;
                if False, the file will be downloaded to memory
        """
        impersonate = self.get_browser_impersonation()

        async with AsyncSession(impersonate=impersonate, timeout=120) as session:
            # Try to determine the size of the file using the url
            response = await session.head(file_url)
            file_type = response.headers.get("Content-Type", "application/octet-stream")
            file_size = int(response.headers.get("Content-Length", -1))

            file_extension = mimetypes.guess_extension(file_type) or ""
            file_extension = file_extension.lstrip(".")

            file_wrapper_params = {
                "file_url": file_url,
                "file_info": FileInfo(
                    file_size=file_size,
                    file_mimetype=file_type,
                    file_extension=file_extension,
                ),
            }

            if file_size == 0:
                raise Exception("File size is 0")

            if force_file_create or file_size == -1 or file_size > self.SIZE_THRESHOLD:
                # Download to file for large/unknown size files
                response = await session.get(file_url, stream=True)
                if response.status_code != 200:
                    raise Exception("File not found")
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    async for chunk in response.aiter_content():
                        tmp.write(chunk)
                    file_wrapper_params["file_path"] = tmp.name
                file_wrapper_params["file_b64_content"] = None
                return FileWrapper(**file_wrapper_params)

            # Download the file to memory
            response = await session.get(file_url)

            if response.status_code != 200:
                raise Exception("File not found")

            b64_content = base64.b64encode(response.content).decode("utf-8")
            file_wrapper_params["file_path"] = None
            file_wrapper_params["file_b64_content"] = b64_content
            return FileWrapper(**file_wrapper_params)
