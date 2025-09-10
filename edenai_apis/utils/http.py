import asyncio
from enum import Enum

import httpx

async_client = httpx.AsyncClient(
    timeout=120,
    limits=httpx.Limits(max_connections=600, max_keepalive_connections=150),
)

client = httpx.Client(
    timeout=120,
    limits=httpx.Limits(max_connections=600, max_keepalive_connections=150),
)


def close():
    asyncio.run(async_client.aclose())
    client.close()


async def aclose():
    await async_client.aclose()
    client.close()


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
