import json
import os
import re
from importlib import import_module
from io import BufferedReader
from typing import Any, Dict, List

from edenai_apis.settings import features_path


def is_valid(x, y):
    return bool(re.search(x, y))


def compare_dicts(dict_a: Dict, dict_b: Dict) -> bool:
    """Return `True` if the two dicts are equivalent, else return `False`"""
    # Check if they have the same keys
    if set(dict_a.keys()) != set(dict_b.keys()):
        return False
    # Compare the content of each field
    for key in dict_a.keys():
        if not compare(dict_a[key], dict_b[key]):
            return False
    # If all OK return True
    return True


def compare_lists(list_a: List[Any], list_b: List[Any]) -> bool:
    """Return `True` if the two lists are equivalent, else return `False`"""
    # Check if they have the same number of elements
    if len(list_a) != len(list_b):
        return False
    # Compare their different elements
    for i, item_a in enumerate(list_a):
        if not compare(item_a, list_b[i]):
            return False
    # If all OK return True
    return True


def compare(items_a: Any, items_b: Any) -> bool:
    """Compare the types of two values"""
    # they must have the same types
    if (
        type_no_int(items_a) != type_no_int(items_b)
        and items_a is not None
        and items_b is not None
    ):
        return False
    # Compare dictionaries
    if isinstance(items_a, dict):
        return compare_dicts(items_a, items_b)
    # Compare lists
    elif isinstance(items_a, list):
        return compare_lists(items_a, items_b)
    # If all OK return True
    return True


def compare_responses(feature: str, subfeature: str, response, phase: str = ""):
    """
    Compare standardized response of a subfeature with the generated output
    Raise `AssertionError` if not equivalent
    Returns `True`
    """
    if phase:
        response_path = os.path.join(
            features_path,
            feature,
            subfeature,
            phase,
            f"{subfeature}_{phase}_response.json",
        )
    else:
        response_path = os.path.join(
            features_path, feature, subfeature, f"{subfeature}_response.json"
        )

    # Some subfeatures have dynamic responses that we can't parse
    # the keys listed in `ignore_keys` won't be compared
    ignore_keys = []
    try:
        key_ignore_function_name = feature + "__" + subfeature + "_ignore"
        subfeature_normalized = subfeature.replace("_async", "")
        imp = import_module(
            f"edenai_apis.features.{feature}.{subfeature_normalized}.ignore_keys"
        )
        ignore_keys = getattr(imp, key_ignore_function_name)()
    except Exception:
        pass

    # Load valid standard response
    with open(response_path, "r", encoding="utf-8") as f:
        standard_response = json.load(f)
        if "original_response" in standard_response:
            raise TypeError(f"Please remove original_response in {response_path}")
        assert_standarization(standard_response, response, ignore_keys=ignore_keys)
        return True


def format_message_error(message: str, path_list_error: list):
    """Format and return path of the non valid value in the asserted object"""
    return message + ". Path: " + ".".join(path_list_error)


def assert_standarization(
    items_a: Any,
    items_b: Any,
    path_list_error: List = None,
    ignore_keys: List = None,
):
    """assert standarization of  a and b"""
    if path_list_error is None:
        path_list_error = ["<root>"]

    if ignore_keys is None:
        ignore_keys = []

    assert_not_none(isinstance(items_a, dict), items_b, path_list_error)

    # if both are not None, check type
    if items_a and items_b:
        # Prevent import MemoryFileUploadHandler
        if not (
            isinstance(items_a, BufferedReader) or isinstance(items_b, BufferedReader)
        ):
            assert (type_no_int(items_a) == type_no_int(items_b)) or issubclass(
                type_no_int(items_b), type_no_int(items_a)
            ), format_message_error(
                f"{type_no_int(items_a).__name__} != {type_no_int(items_b).__name__}",
                path_list_error,
            )

    # if both are list
    if isinstance(items_a, list) or isinstance(items_b, list):
        assert_equivalent_list(items_a, items_b, path_list_error, ignore_keys)

    # if both are dict
    elif isinstance(items_a, dict) or isinstance(items_b, dict):
        assert_equivalent_dict(items_a, items_b, path_list_error, ignore_keys)


def assert_equivalent_list(
    list_a: List, list_b: List, path_list_error: list, ignore_keys: list
):
    """Assert List `a` and `b` are equivalent"""
    # check both are list
    assert isinstance(list_a, list) and isinstance(list_b, list), format_message_error(
        "Not two lists", path_list_error
    )

    # check both are not empty and check first element
    if len(list_a) > 0 and len(list_b) > 0:
        if isinstance(list_b[0], dict):
            assert_equivalent_dict(
                list_a[0], list_b[0], path_list_error + ["0"], ignore_keys
            )
        elif isinstance(list_b[0], list):
            assert_equivalent_list(
                list_a[0], list_b[0], path_list_error + ["0"], ignore_keys
            )


def assert_equivalent_dict(
    dict_a: Dict,
    dict_b: Dict,
    path_list_error: list = None,
    ignore_keys: list = None,
):
    """Assert Dict `a` and `b` are equivalent"""
    if path_list_error is None:
        path_list_error = []

    if ignore_keys is None:
        ignore_keys = []

    # check both are dict
    assert isinstance(dict_a, dict) and isinstance(dict_b, dict), format_message_error(
        "Not two dicts", path_list_error
    )

    # check dicts have same keys
    assert_list_unordered_equality(
        list(dict_a.keys()), list(dict_b.keys()), path_list_error, "keys"
    )

    # check all keys
    for key in dict_a:
        if key not in ignore_keys:
            key_a = dict_a.get(key)
            key_b = dict_b.get(key)

            # check value of key for both dict
            assert_standarization(key_a, key_b, path_list_error + [key], ignore_keys)


def assert_list_unordered_equality(
    list_a: List, list_b: List, path_list_error: List, str_type
):
    """check list equality without order"""
    lacks = sorted(list(set(list_a) - set(list_b)))
    assert not lacks, format_message_error(
        f"Missing {str_type} {lacks}", path_list_error
    )
    extra = sorted(list(set(list_b) - set(list_a)))
    assert not extra, format_message_error(f"Extra {str_type} {extra}", path_list_error)


def assert_not_none(items_a: Any, items_b: Any, path_list_error: list):
    """check if one is None that other is not list or dict"""
    if items_a is None:
        assert not isinstance(items_b, list), format_message_error(
            "None and dict", path_list_error
        )
        assert not isinstance(items_b, dict), format_message_error(
            "None and list", path_list_error
        )
    if items_b is None:
        assert not isinstance(items_a, list), format_message_error(
            "None and dict", path_list_error
        )
        assert not isinstance(items_a, dict), format_message_error(
            "None and list", path_list_error
        )


def type_no_int(var):
    """Convert to `float` if var is of type `int`"""
    return type(var) if not isinstance(var, int) else float
