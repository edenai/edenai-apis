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


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
