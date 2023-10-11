import logging
from typing import Any, List, Optional, TypeVar, Union

T = TypeVar("T")

def extract(
    obj: Union[dict, list],
    path: List[Union[str, int]],
    fallback: Optional[T] = None,
    type_validator: Optional[type] = None,
) -> Union[Any, T, None]:
    """
    Extract a value from a dict given a path & optional fallback + type validator

    Args:
        obj (dict): object to extract value from
        path (List[str | int]): list of keys/indexed representing the path of the value to extract (eg: obj['result'][0] will be ['result', 0])
        fallback (Any), optional: fallback value if something goes wrong while trying to get extract value
        type_validator (type), optional: expected type of extracted value, if they are different, we return the fallback

    Returns:
        Any: extracted value of Any type or fallback type

    Example:
      Extract a value from an object
      >>> obj = {"one": {"two": [0, "result", 0]}}
      >>> extract(obj, ["one", "two", 1])
      "result"

      Get fallback Value if extraction failed
      >>> obj = {"one": {"two": [0, "result", 0]}}
      >>> extract(obj, ["one", "two", 3], fallback="FALLBACK_VALUE")
      "FALLBACK_VALUE"

      Simple type Validation, return fallback if validation fail
      >>> obj = {"one": {"two": [0, "result", 0]}}
      >>> extract(obj, ["one", "two", 3], fallback="FALLBACK_VALUE", type_validator=int)
      "FALLBACK_VALUE"
    """
    result = obj
    try:
        for key in path:
            result = result[key]  # type: ignore
    except (KeyError, IndexError, TypeError) as exc:
        logging.error(
            f"{exc.__class__.__name__}: {exc} while trying to extract {path} of object {obj}, returning fallback: {fallback}"
        )
        return fallback

    if type_validator is not None and type(result) != type_validator:
        logging.error(
            f"Object {result} of type {type(result)} is not of expected type: {type_validator}, returning fallback: {fallback}"
        )
        return fallback

    return result
