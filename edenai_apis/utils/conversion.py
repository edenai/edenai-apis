import re
import locale
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from edenai_apis.utils.public_enum import AutomlClassificationProviderName


def _format_string_for_conversion(string_number: str) -> str:
    commas_occurences = [match.start() for match in re.finditer("\,", string_number)]
    dot_occurences = [match.start() for match in re.finditer("\.", string_number)]

    if len(commas_occurences) > 0 and len(dot_occurences) > 0:
        index_remove_partt = max(
            commas_occurences[len(commas_occurences) - 1],
            dot_occurences[len(dot_occurences) - 1],
        )
        number_part = string_number[:index_remove_partt]
        degit_part = string_number[index_remove_partt + 1 :]
        number_part = re.sub(r"[^\d]", "", number_part)
        return f"{number_part}.{degit_part}"
    if len(commas_occurences) > 0:
        if len(commas_occurences) == 1:
            return string_number.replace(",", ".")
    if len(dot_occurences) > 0:
        if len(dot_occurences) == 1:
            return string_number
    return re.sub(r"[^\d]", "", string_number)


def convert_string_to_number(
    string_number: Optional[str], val_type: Union[Type[int], Type[float]]
) -> Union[int, float, None]:
    """convert a `string` to either `int` or `float`"""
    if not string_number:
        return None
    if isinstance(string_number, (int, float)):
        return string_number
    if isinstance(string_number, str):
        string_number = string_number.strip()
    try:
        number_nature = 1
        # test if negatif element
        if string_number[0] == "-":
            number_nature = -1
        string_formatted = _format_string_for_conversion(
            re.sub(r"[^\d\.\,]", "", string_number)
        )
        return val_type(float(string_formatted)) * number_nature
    except Exception as exc:
        print(exc)
        return None


def closest_above_value(input_list, input_value):
    try:
        above = min(
            [i for i in input_list if i >= input_value] or input_list,
            key=lambda x: abs(x - input_value),
        )
    except ValueError:
        above = -1
    return above


def closest_below_value(input_list, input_value):
    try:
        below = min(
            [i for i in input_list if i <= input_value] or input_list,
            key=lambda x: abs(x - input_value),
        )
    except ValueError:
        below = -1
    return below


def standardized_confidence_score(
    confidence_score: float,
    limit_values: list = [0.2, 0.4, 0.6, 0.8, 1.0],
    ratio: float = 5,
):
    return closest_above_value(limit_values, confidence_score) * ratio


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


def combine_date_with_time(
    date: Optional[str], time: Union[str, None]
) -> Union[str, None]:
    """
    Concatenate date string and time string
    Returns:
        - `None`: if `date` is None
        - `str`: if concatenation is successful or if `date` exist
    """
    if time and date:
        for fmt in ["%H:%M", "%H:%M:%S"]:
            try:
                time = dt.datetime.strptime(str(time), fmt).time()
                date = str(
                    dt.datetime.combine(dt.datetime.strptime(date, "%Y-%m-%d"), time)
                )
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
    ptdate_regex = re.compile(
        r"PT(?:(?P<hours>([0-9]*[.])?[0-9]+)H)?(?:(?P<minutes>([0-9]*[.])?[0-9]+)M)?(?:(?P<seconds>([0-9]*[.])?[0-9]+)S)?"
    )
    match = ptdate_regex.match(pt_date)
    if match:
        hours = float(match.group("hours") or 0)
        minutes = float(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        return 3600 * hours + 60 * minutes + seconds
    return None


# Lambda define for more accurate in add_query_param_in_url function
is_first_param_in_url = lambda url: "?" not in url


def add_query_param_in_url(url: str, query_params: dict):
    if query_params and url:
        for key, value in query_params.items():
            if not key or not value:
                continue
            url = (
                f"{url}?{key}={value}"
                if is_first_param_in_url(url)
                else f"{url}&{key}={value}"
            )
    return url


def concatenate_params_in_url(url: str, params: list, sep: str):
    if params and url:
        for param in params:
            if not param:
                continue
            url += sep + param
    return url


def find_all_occurrence(a_str, sub):
    if sub is None or len(sub) == 0:
        return []
    start = 0
    result = []
    while True:
        start = a_str.find(sub, start)

        if start == -1:
            return result

        result.append(start)
        start += len(sub)


def replace_sep(x: str, current_sep: str, new_sep: str):
    if isinstance(x, str):
        x = x.replace(current_sep, new_sep)
        x = re.sub(r"{}$".format(re.escape(new_sep)), "", x)
    return x


def convert_pitch_from_percentage_to_semitones(speaking_pitch: float):
    # we assume that the biggest shift is 12 (-/+) semitones
    # 1 demi-ton/seconde mineure : 1,059463   (pow(2, n/12))
    # 2 demi-tons/seconde majeure : 1,122462
    # 3 demi-tons/tierce mineure : 1.189207
    # 4 demi-tons/tierce majeure : 1.259921
    # 5 demi-tons/quarte parfaite : 1,334840
    # 6 demi-tons/triton : 1.414214
    # 7 demi-tons/quinte juste : 1,498307
    # 8 demi-tons/sixte mineure : 1,587401
    # 9 demi-tons/sixte majeure : 1,681793
    # 10 demi-tons/septième mineure : 1,781797
    # 11 demi-tons/septième majeure : 1,887749
    # 12 demi-tons/octave : 2,0

    semitones = [
        1.059463,
        1.122462,
        1.189207,
        1.259921,
        1.334840,
        1.414214,
        1,
        587401,
        1.681793,
        1,
        781797,
        1.887749,
        2.0,
    ]

    if speaking_pitch > 100:
        speaking_pitch = 100
    if speaking_pitch < -100:
        speaking_pitch = -100
    sign = 1 if speaking_pitch > 0 else -1
    diff = abs(speaking_pitch) / 100
    shifting = 1 + diff
    semitone = min(semitones, key=lambda x: abs(x - shifting))
    return sign * semitones.index(semitone)


def standardized_confidence_score_picpurify(confidence_score: float, nsfw: bool):
    if nsfw:
        if confidence_score >= 0.8:
            return 5
        elif confidence_score >= 0.6:
            return 4
        elif confidence_score >= 0.4:
            return 3
        elif confidence_score >= 0.2:
            return 2
    else:
        return 1


def construct_word_list(original_text, corrected_words):
    word_list = []
    for correction in corrected_words:
        word_with_mistake = correction["word"]
        corrected_word = correction["correction"]

        # Find the index of the word with the mistake in the original text
        offset = original_text.find(word_with_mistake)
        real_offset = closest_above_value(
            find_all_occurrence(original_text, word_with_mistake), offset
        )
        length = len(word_with_mistake)

        # Create a new dictionary with the extracted information
        word_info = {
            "word": word_with_mistake,
            "offset": real_offset,
            "length": length,
            "suggestion": corrected_word,
        }

        # Append to the final list
        word_list.append(word_info)

    return word_list


def iterate_all(iterable, returned="key"):
    """Returns an iterator that returns all keys or values
    of a (nested) iterable.

    Arguments:
        - iterable: <list> or <dictionary>
        - returned: <string> "key" or "value"

    Returns:
        - <iterator>
    """
    if isinstance(iterable, dict):
        for key, value in iterable.items():
            if returned == "key":
                yield key
            elif returned == "value":
                if not (isinstance(value, dict) or isinstance(value, list)):
                    yield value
            else:
                raise ValueError("'returned' keyword only accepts 'key' or 'value'.")
            for ret in iterate_all(value, returned=returned):
                yield ret
    elif isinstance(iterable, list):
        for el in iterable:
            for ret in iterate_all(el, returned=returned):
                yield ret
    yield iterable
