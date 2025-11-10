import asyncio
import base64
from io import BytesIO
import uuid

from edenai_apis.utils.upload_s3 import USER_PROCESS, aupload_file_bytes_to_s3


async def aget_resource_url(image_b64: str, image_format: str) -> dict[str, str]:
    """
    Upload image to S3 asynchronously.

    Arguments:
        image_b64 (str): Base64 encoded image.
        image_format (str): Image format (e.g., 'png', 'jpg').

    Returns:
        dict[str, str]: Dictionary containing the base64 encoded image and the resource URL.
    """

    # Decode base64 in thread pool (CPU-bound operation)
    def decode_image():
        base64_bytes = image_b64.encode("ascii")
        return BytesIO(base64.b64decode(base64_bytes))

    image_bytes = await asyncio.to_thread(decode_image)

    # Upload to S3 asynchronously
    resource_url = await aupload_file_bytes_to_s3(
        image_bytes, image_format, USER_PROCESS
    )

    return {"image_b64": image_b64, "image_resource_url": resource_url}
