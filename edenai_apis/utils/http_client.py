from contextlib import asynccontextmanager
from typing import Union

import httpx


# Timeout presets (connect=10s, varying read timeouts)
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=120.0)  # General purpose
OCR_TIMEOUT = httpx.Timeout(10.0, read=180.0)  # OCR/document parsing (can be slow)
IMAGE_TIMEOUT = httpx.Timeout(10.0, read=180.0)  # Image generation/processing
QUICK_TIMEOUT = httpx.Timeout(10.0, read=30.0)  # Fast API calls
AUDIO_TIMEOUT = httpx.Timeout(10.0, read=320.0)  # Audio processing


@asynccontextmanager
async def async_client(timeout: Union[httpx.Timeout, float] = DEFAULT_TIMEOUT, **kwargs):
    """
    Create an httpx AsyncClient with standardized configuration.

    Usage:
        from edenai_apis.utils.http_client import async_client, OCR_TIMEOUT

        async with async_client(OCR_TIMEOUT) as client:
            response = await client.post(url, ...)

    Args:
        timeout: Timeout configuration. Use presets (DEFAULT_TIMEOUT, OCR_TIMEOUT, etc.)
                 or pass a float for simple read timeout with 10s connect.
        **kwargs: Additional arguments passed to httpx.AsyncClient
    """
    if isinstance(timeout, (int, float)):
        timeout = httpx.Timeout(10.0, read=float(timeout))

    async with httpx.AsyncClient(timeout=timeout, **kwargs) as client:
        yield client
