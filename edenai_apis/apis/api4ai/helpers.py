from typing import Optional, Union

import httpx
from requests import Response


def get_errors_from_response(
    response: Union[Response, httpx.Response],
) -> Optional[str]:
    """
    Return formated error from a requests.Response object if it exists,
    else return None
    """
    if response.status_code == 200:
        data = response.json()
        status = data["results"][0].get("status", {})
        if status.get("code") == "failure":
            return status.get("message")
        else:
            return

    elif response.status_code == 413:
        return (
            "Your client issued a request that was too large. "
            "Please do not send a file larger than 16mb."
        )
    elif response.status_code == 422:
        detail = response.json()["detail"][0] or {}
        return f"{detail['msg']} for {', '.join(detail['loc'])}"

    elif response.status_code in [400, 401]:
        return response.json()["message"]

    else:
        return response.text
