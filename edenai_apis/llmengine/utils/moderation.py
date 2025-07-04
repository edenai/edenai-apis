from typing import Union, List
from enum import Enum

from functools import wraps
from asgiref.sync import async_to_sync
import asyncio
import aiohttp


from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException


class OpenAIErrorCode(Enum):
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_IMAGE = "invalid_image_format"
    INVALID_IMAGE_URL = "invalid_data_url"


async def moderate_content(headers, content: Union[str, List]) -> bool:
    if not content:
        return False

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/moderations",
            headers=headers,
            json={"model": "omni-moderation-latest", "input": content},
        ) as response:
            response_data = await get_openapi_response_async(response)

            if response_data is None:
                return False

            flagged = response_data["results"][0]["flagged"]

            if flagged:
                categories = [
                    category
                    for category, value in response_data["results"][0][
                        "categories"
                    ].items()
                    if value
                ]
                message = f"Content rejected due to the violation of the following policies: {', '.join(categories)}."
                raise ProviderException(message=message, code=400)

    return not flagged


async def standard_moderation(*args, **kwargs):
    api_settings = load_provider(ProviderDataEnum.KEY, "openai", api_keys={})

    api_key = api_settings.get("api_key")
    headers = {"Authorization": f"Bearer {api_key}"}

    tasks = []

    messages = kwargs.get("messages", [])

    for message in messages:
        if "content" in message:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    tasks.append(moderate_content(headers, [content]))
            else:
                tasks.append(moderate_content(headers, message["content"]))

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*tasks)


async def moderate_if_exists(headers, value):
    if value:
        await moderate_content(headers, value)  # could be a problem


async def get_openapi_response_async(response: aiohttp.ClientResponse):
    """
    This function takes an aiohttp.ClientResponse as input and returns its response.json()
    raises a ProviderException if the response contains an error.
    """
    try:
        original_response = await response.json()
        if "error" in original_response or response.status >= 400:
            code = original_response["error"]["code"]
            if code == OpenAIErrorCode.RATE_LIMIT_EXCEEDED.value:
                return None
            if code == OpenAIErrorCode.INVALID_IMAGE.value:
                return None
            if code == OpenAIErrorCode.INVALID_IMAGE_URL.value:
                return None
            message_error = original_response["error"]["message"]
            raise ProviderException(message_error, code=response.status)
        return original_response
    except Exception:
        raise ProviderException(await response.text(), code=response.status)


def prepare_messages_for_moderation(headers, content):
    content_data = content["content"]

    if "text" in content_data:
        return moderate_if_exists(headers, content_data.get("text"))

    elif "media_url" in content_data:
        image_data = [
            {"type": "image_url", "image_url": {"url": content_data["media_url"]}}
        ]
        return moderate_if_exists(headers, image_data)

    elif "media_base64" in content_data:
        image_data = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{content_data['media_base64']}"
                },
            }
        ]
        return moderate_if_exists(headers, image_data)


async def check_content_moderation_async(*args, **kwargs):
    api_settings = load_provider(ProviderDataEnum.KEY, "openai", api_keys={})

    api_key = api_settings.get("api_key")
    headers = {"Authorization": f"Bearer {api_key}"}
    tasks = []

    tasks.append(moderate_if_exists(headers, kwargs.get("text")))
    tasks.append(moderate_if_exists(headers, kwargs.get("chatbot_global_action")))
    tasks.append(moderate_if_exists(headers, kwargs.get("instruction")))

    if kwargs.get("previous_history"):
        tasks.extend(
            moderate_if_exists(headers, item.get("message"))
            for item in kwargs["previous_history"]
            if isinstance(item, dict)
        )

    if "texts" in kwargs:
        tasks.extend(moderate_if_exists(headers, item) for item in kwargs["texts"])

    if "image_data" in kwargs:
        tasks.append(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": kwargs.get("image_data")},
                }
            ]
        )

    if "messages" in kwargs:
        for message in kwargs["messages"]:
            if isinstance(message, dict) and "content" in message:
                for content in message["content"]:
                    if isinstance(content, dict) and "content" in content:
                        tasks.append(prepare_messages_for_moderation(headers, content))
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*tasks)


def check_content_moderation(*args, **kwargs):
    async_to_sync(check_content_moderation_async)(*args, **kwargs)


def moderate(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if kwargs.get("moderate_content"):
            check_content_moderation(*args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def async_moderate(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if kwargs.get("moderate_content"):
            await check_content_moderation_async(*args, **kwargs)
        return await func(self, *args, **kwargs)

    return wrapper


def moderate_std(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if kwargs.get("moderate_content"):
            async_to_sync(standard_moderation)(*args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def async_moderate_std(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if kwargs.get("moderate_content"):
            await standard_moderation(*args, **kwargs)
        return await func(self, *args, **kwargs)

    return wrapper
