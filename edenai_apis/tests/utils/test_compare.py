"""
Test functions used to compare output dictionaries.
"""

import pytest

from edenai_apis.utils.compare import assert_standarization, compare

a = {"a": [1, 2, 3], "b": {"e": "one", "f": "two"}, "c": 3}

b = {"a": [1, 2, 3], "c": 43, "b": {"e": "hello", "f": "world"}}

c = {"a": [1, 2, 3], "c": 43, "b": {"e": "hello", "f": "world"}, "e": 1}

d = {"a": [1, 2, 3, 4], "c": 43, "b": {"e": "hello", "f": "world"}}

e = {"a": [1, 2, 3, 4], "c": 43, "b": {"e": "hello", "f": 1}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "first,second,expected",
    [
        [a, b, True],
        [a, c, False],
        [a, d, False],
        [a, e, False],
    ],
)
def test_compare(first, second, expected):
    """Test function compare"""
    assert compare(first, second) == expected
    assert compare(second, first) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "first,second,expected_assertion_exception",
    [
        [a, b, None],
        [a, c, "Extra keys ['e']. Path: <root>"],
        [a, d, None],
        [a, e, "str != float. Path: <root>.b.f"],
        [a, {"c": 3}, "Missing keys ['a', 'b']. Path: <root>"],
        [{"c": 3}, {"c": 3.5}, None],
        [{"c": 3}, {"c": None}, None],
        [{"a": [{"b": 1}]}, {"a": [{"b": 2}]}, None],
        [{"a": [{"b": 1}]}, {"a": [{"b": 3}, {"b": 3}]}, None],
        [
            {"a": [{"b": 1}]},
            {"a": [{"c": 3}, {"b": 3}]},
            "Missing keys ['b']. Path: <root>.a.0",
        ],
        [{"a": [{"b": 1}]}, {"a": [{"b": 3}, {"c": 3}]}, None],
    ],
)
def test_compare_standarization(first, second, expected_assertion_exception):
    """Test function assert_standarization"""
    if expected_assertion_exception:
        with pytest.raises(AssertionError) as assertion_exception:
            assert_standarization(first, second)
        # these asserts are identical; you can use either one
        assert str(assertion_exception.value) == expected_assertion_exception

    else:
        assert_standarization(first, second)
