import base64
import tempfile
from edenai_apis.utils.files import FileInfo, FileWrapper
from edenai_apis.utils.http import async_client


class FileHandler:

    SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 Mb

    def __init__(self):
        pass

    async def download_file(self, file_url: str) -> FileWrapper:
        """
        Download a file from a url and return the file path

        Args:
            file_url (str): url of the file to download
            if False, the file will be downloaded to memory
        """
        # wrapper = FileWrapper()
        # Try to determine the ize of the file suing the url, fallback if not possible
        response = await async_client.head(file_url)
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
            async with async_client.stream("GET", file_url) as stream:
                if stream.status_code != 200:
                    raise Exception("File not found")
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    async for chunk in stream.aiter_bytes():
                        tmp.write(chunk)
                    file_wrapper_params["file_path"] = tmp.name
            file_wrapper_params["file_b64_content"] = None
            return FileWrapper(**file_wrapper_params)

        # Download the file
        response = await async_client.get(file_url)

        if response.status_code != 200:
            raise Exception("File not found")

        b64_content = base64.b64encode(response.content).decode("utf-8")
        file_wrapper_params["file_path"] = None
        file_wrapper_params["file_b64_content"] = b64_content
        return FileWrapper(**file_wrapper_params)
