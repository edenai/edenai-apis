import asyncio
import logging
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

async_client = httpx.AsyncClient(
    timeout=120,
    limits=httpx.Limits(max_connections=600, max_keepalive_connections=150),
)

client = httpx.Client(
    timeout=120,
    limits=httpx.Limits(max_connections=600, max_keepalive_connections=150),
)


def close():
    """Synchronously close both sync and async httpx clients."""
    try:
        if not async_client.is_closed:
            asyncio.run(async_client.aclose())
    except RuntimeError as exc:
        # We're in an async event loop: can't use asyncio.run
        raise RuntimeError(
            f"Couldn't close async_client: {exc}. "
            "Please use `aclose` if you are running in an event loop."
        )
    finally:
        if not client.is_closed:
            client.close()
    logger.info("Successfully closed sync & async httpx clients.")


async def aclose():
    """Asynchronously close both sync and async httpx clients."""
    if not async_client.is_closed:
        await async_client.aclose()
    if not client.is_closed:
        client.close()
    logger.info("Successfully closed sync & async httpx clients.")


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
