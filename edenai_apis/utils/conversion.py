import re
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple, Union

from edenai_apis.utils.public_enum import AutomlClassificationProviderName


def convert_string_to_number(
    string_number: Optional[str],
    val_type: Union[int, float]
) -> Union[int, float, None]:
    """convert a `string` to either `int` or `float`"""
    if not string_number:
        return None
    if isinstance(string_number, (int, float)):
        return string_number
    if isinstance(string_number, str):
        string_number = string_number.strip()
    try:
        number = val_type(re.sub(r"[^\d\.]", "", string_number))
        return number
    except Exception as exc:
        return None


def retreive_first_number_from_string(string_number: str) -> Union[str, None]:
    """
    Find the first number found in a string
    Returns:
        str:    if found the number is returned as a string
        None:   if nothing is found
    """
    if isinstance(string_number, str) and string_number:
        numbers = re.findall(r"\d+", string_number)
        return numbers[0] if numbers else None
    return None


def combine_date_with_time(date: Optional[str], time: Union[str, None]) -> Union[str, None]:
    """
    Concatenate date string and time string
    Returns:
        - `None`: if `date` is None
        - `str`: if concatenation is successful or if `date` exist
    """
    if time and date:
        for fmt in ["%H:%M", "%H:%M:%S"]:
            try:
                time = dt.datetime.strptime(time, fmt).time()
                date = str(dt.datetime.combine(dt.datetime.strptime(date, "%Y-%m-%d"), time))
                break
            except ValueError as exc:
                continue
    return date

# TODO test it
def from_jsonarray_to_list(
        class_ref, json_list: List[Dict], list_tuples_json_class: List[Tuple[Any, Any]]
    ) -> List:
        """
        Transforme un json_list (
            [
                {k1 : V1, K2 : V2},
                {k3 : V3, K4 : V4}
            ]) en une list[ClassName] python
        Args:
            - `json_list`: List of Json à transformer
            - `ClassName`: qui est le nom de la class dont sera fait la list
            - `list_tuples_json_class`: une list de tuples pour matcher chaque key du json object
        avec l'attribue qui lui correspond dans la class ClassName : [(k1, attr1), (k2, attr2)]
        """
        list_instances: List[class_ref] = []
        attrs = [
            attr
            for attr in dir(class_ref)
            if not callable(getattr(class_ref, attr)) and not attr.startswith("__")
        ]
        tmp = {}
        for json_obj in json_list:
            for item in list_tuples_json_class:
                if item[1] not in attrs:
                    pass
                tmp[item[1]] = json_obj[item[0]]
            instance = class_ref(**tmp)
            list_instances.append(instance)

        return list_instances


def convert_pt_date_from_string(pt_date: str) -> int:
    if not isinstance(pt_date, str):
        return None
    ptdate_regex = re.compile(r"PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?")
    match = ptdate_regex.match(pt_date)
    if match:
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)
        return 3600 * hours + 60 * minutes + seconds
    return None

# Lambda define for more accurate in add_query_param_in_url function
is_first_param_in_url = lambda url: '?' not in url

def add_query_param_in_url(url: str, query_params: dict):
    if query_params and url:
        for key, value in query_params.items():
            if not key or not value:
                continue
            url = f'{url}?{key}={value}' if is_first_param_in_url(url) else f'{url}&{key}={value}'
    return url

def concatenate_params_in_url(url: str, params: list, sep: str):
    if params and url:
        for param in params:
            if not param:
                continue
            url += sep + param
    return url


def replace_sep(x: str, current_sep: str, new_sep: str):
    if isinstance(x, str):
        x = x.replace(current_sep, new_sep)
        x = re.sub(r"{}$".format(re.escape(new_sep)), "", x)
    return x
