import base64
import random
import tempfile
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


class FileHandler:

    SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 Mb

    def __init__(self):
        pass

    @staticmethod
    def get_user_agent():
        random_ua = random.choice(USER_AGENTS)
        return {"User-Agent": random_ua}

    async def download_file(self, file_url: str) -> FileWrapper:
        """
        Download a file from a url and return the file path

        Args:
            file_url (str): url of the file to download
            if False, the file will be downloaded to memory
        """
        # Try to determine the size of the file using the url, fallback if not possible
        response = await async_client.head(
            file_url, headers=FileHandler.get_user_agent()
        )
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
        if file_size == -1 or file_size > self.SIZE_THRESHOLD:
            # The file will be downloaded lazily when the b64 content is requested
            async with async_client.stream(
                "GET", file_url, headers=FileHandler.get_user_agent()
            ) as stream:
                if stream.status_code != 200:
                    raise Exception("File not found")
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    async for chunk in stream.aiter_bytes():
                        tmp.write(chunk)
                    file_wrapper_params["file_path"] = tmp.name
            file_wrapper_params["file_b64_content"] = None
            return FileWrapper(**file_wrapper_params)

        # Download the file
        response = await async_client.get(
            file_url, headers=FileHandler.get_user_agent()
        )

        if response.status_code != 200:
            raise Exception("File not found")

        b64_content = base64.b64encode(response.content).decode("utf-8")
        file_wrapper_params["file_path"] = None
        file_wrapper_params["file_b64_content"] = b64_content
        return FileWrapper(**file_wrapper_params)
