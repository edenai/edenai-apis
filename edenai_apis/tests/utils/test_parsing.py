from edenai_apis.utils.parsing import extract


def test_parsing_extract_dict():
    expected_output = "yay"
    test_obj = {"first_level": {"second_level": {"third_level": expected_output}}}

    test_output = extract(test_obj, ["first_level", "second_level", "third_level"])

    assert test_output == expected_output


def test_parsing_extract_list():
    expected_output = "yay"
    test_obj = [0, [0, 0, 0, [0, 0, expected_output]]]

    test_output = extract(test_obj, [1, 3, 2])

    assert test_output == expected_output


def test_parsing_extract_mixed_object():
    expected_output = "yay"
    test_obj = {"first_level": [{0: 0}, {"second": expected_output}]}

    test_output = extract(test_obj, ["first_level", 1, "second"])

    assert test_output == expected_output


def test_parsing_fallback_on_index_error():
    expected_output = "yay"
    test_obj = [0, 0, 0, 0]

    test_output = extract(test_obj, [4], fallback=expected_output)

    assert test_output == expected_output


def test_parsing_fallback_on_key_error():
    expected_output = "yay"
    test_obj = {"hello": "world"}

    test_output = extract(test_obj, ["_"], fallback=expected_output)

    assert test_output == expected_output


def test_parsing_fallback_on_type_error():
    expected_output = "yay"
    test_obj = {"first_level": None}

    test_output = extract(
        test_obj, ["first_level", "second_level"], fallback=expected_output
    )

    assert test_output == expected_output


def test_parsing_fallback_with_type_validation():
    expected_output = {"second_level": 0}
    test_obj = {"first_level": expected_output}

    test_output = extract(
        test_obj, ["first_level"], fallback=expected_output, type_validator=dict
    )

    assert test_output == expected_output


def test_parsing_type_validation_error():
    expected_output = "yay"
    test_obj = {"first_level": 1}

    test_output = extract(
        test_obj, ["first_level"], fallback=expected_output, type_validator=str
    )

    assert test_output == expected_output
