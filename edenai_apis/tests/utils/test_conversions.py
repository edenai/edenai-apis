from typing import Generator

import pytest

from edenai_apis.utils.conversion import (
    add_query_param_in_url,
    closest_above_value,
    closest_below_value,
    combine_date_with_time,
    concatenate_params_in_url,
    convert_pt_date_from_string,
    convert_string_to_number,
    iterate_all,
    replace_sep,
    retreive_first_number_from_string,
)


class TestConvertStringToNumber:
    @pytest.mark.unit
    def test_string_to_int(self):
        string_number = "1234"
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = 1234
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_to_float(self):
        string_number = "12.34"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12.34
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_non_numeric_chars(self):
        string_number = "12.34abc"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12.34
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_one_comma(self):
        string_number = "12,34"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12.34
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_multiple_commas(self):
        string_number = "12,345,123"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12345123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_multiple_dots(self):
        string_number = "12.345.123"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12345123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_commas_and_dots_finished_with_comma(self):
        string_number = "12.345,123"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12345.123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_commas_and_dots_finished_with_dot(self):
        string_number = "12,345.123"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 12345.123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_negatif_number_to_float(self):
        string_number = "-12,345.123"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = -12345.123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_negatif_number_to_int(self):
        string_number = "-1234"
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = -1234
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_empty_string(self):
        string_number = ""
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_sign(self):
        string_number = "-"
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_none_value(self):
        string_number = None
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_int_value(self):
        string_number = 123
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = 123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_float_value(self):
        string_number = 123.45
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 123.45
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_float_string_to_int(self):
        string_number = "123.45"
        val_type = int
        output = convert_string_to_number(string_number, val_type)
        expected_output = 123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"

    @pytest.mark.unit
    def test_int_string_to_float(self):
        string_number = "123"
        val_type = float
        output = convert_string_to_number(string_number, val_type)
        expected_output = 123
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({string_number}, {val_type}) but got `{output}`"


class TestRetrieveFirstNumberFromString:
    @pytest.mark.unit
    def test_valid_number_with_one_number(self):
        string_number = "Bonjour1"
        output = retreive_first_number_from_string(string_number)
        expected_output = "1"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{string_number}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_with_two_number_in_a_row(self):
        string_number = "Bonjour12"
        output = retreive_first_number_from_string(string_number)
        expected_output = "12"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{string_number}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_with_two_number_not_in_a_row(self):
        string_number = "Bonjour1q2"
        output = retreive_first_number_from_string(string_number)
        expected_output = "1"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{string_number}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_without_number(self):
        string_number = "Bonjour"
        output = retreive_first_number_from_string(string_number)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{string_number}` but got `{output}`"

    @pytest.mark.unit
    def test_empty_string(self):
        string_number = ""
        output = retreive_first_number_from_string(string_number)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{string_number}` but got `{output}`"

    @pytest.mark.unit
    def test_int_input(self):
        string_number = 2
        output = retreive_first_number_from_string(string_number)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{string_number}` but got `{output}`"


class TestCombineDateWithTime:
    @pytest.mark.unit
    def test_valid_date_and_time(self):
        date = "2022-01-01"
        time = "12:34:56"
        output = combine_date_with_time(date, time)
        expected_output = "2022-01-01 12:34:56"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({date}, {time}) but got `{output}"

    @pytest.mark.unit
    def test_valid_date_and_time_without_seconds(self):
        date = "2022-01-01"
        time = "12:34"
        output = combine_date_with_time(date, time)
        expected_output = "2022-01-01 12:34:00"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({date}, {time}) but got `{output}"

    @pytest.mark.unit
    def test_valid_date_and_time_with_invalid_format(self):
        date = "2022-01-01"
        time = "12:34:56:78"
        output = combine_date_with_time(date, time)
        expected_output = "2022-01-01"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({date}, {time}) but got `{output}"

    @pytest.mark.unit
    def test_valid_date_and_time_with_invalid_time(self):
        date = "2022-01-01"
        time = "12:34:56:78:90"
        output = combine_date_with_time(date, time)
        expected_output = "2022-01-01"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({date}, {time}) but got `{output}"

    @pytest.mark.unit
    def test_valid_date_and_no_time(self):
        date = "2022-01-01"
        time = None
        output = combine_date_with_time(date, time)
        expected_output = "2022-01-01"
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({date}, {time}) but got `{output}"

    @pytest.mark.unit
    def test_no_date_and_valid_time(self):
        date = None
        time = "12:34:56"
        output = combine_date_with_time(date, time)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({date}, {time}) but got `{output}"


class TestConvertPtDateToString:
    @pytest.mark.unit
    def test_valid_pt_date_with_hours_minutes_seconds(self):
        pt_date = "PT1H1M10S"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 3670
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_valid_pt_date_with_hours_and_seconds(self):
        pt_date = "PT1H10S"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 3610
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_valid_pt_date_with_hours_and_minutes(self):
        pt_date = "PT1H10S"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 3610
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_valid_pt_date_with_hours(self):
        pt_date = "PT10H"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 36000
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_valid_pt_date_with_minutes(self):
        pt_date = "PT10M"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 600
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_valid_pt_date_with_seconds(self):
        pt_date = "PT10S"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 10
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_valid_pt_date_with_minutes_and_seconds(self):
        pt_date = "PT10M10S"
        output = convert_pt_date_from_string(pt_date)
        expected_output = 610
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_invalid_string_pt_date(self):
        pt_date = "Invalid Format"
        output = convert_pt_date_from_string(pt_date)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"

    @pytest.mark.unit
    def test_invalid_none_pt_date(self):
        pt_date = None
        output = convert_pt_date_from_string(pt_date)
        expected_output = None
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{pt_date}` but got `{output}"


class TestAddQueryParamInUrl:
    @pytest.mark.unit
    def test_add_query_param_in_url_empty_query_string(self):
        url = "http://www.example.com"
        query_params = {"param1": "value1", "param2": "value2"}
        expected_url = "http://www.example.com?param1=value1&param2=value2"
        output_url = add_query_param_in_url(url, query_params)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={query_params}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_query_param_in_url_existing_query_string(self):
        url = "http://www.example.com?param3=value3"
        query_params = {"param1": "value1", "param2": "value2"}
        expected_url = (
            "http://www.example.com?param3=value3&param1=value1&param2=value2"
        )
        output_url = add_query_param_in_url(url, query_params)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={query_params}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_query_param_in_url_none_value(self):
        url = "http://www.example.com"
        query_params = {"param1": "value1", "param2": None}
        expected_url = "http://www.example.com?param1=value1"
        output_url = add_query_param_in_url(url, query_params)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={query_params}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_query_param_in_url_empty_query_param(self):
        url = "http://www.example.com"
        query_params = {}
        expected_url = "http://www.example.com"
        output_url = add_query_param_in_url(url, query_params)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={query_params}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_none_param_in_url_empty_query_param(self):
        url = "http://www.example.com"
        query_params = None
        expected_url = "http://www.example.com"
        output_url = add_query_param_in_url(url, query_params)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={query_params}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_query_param_in_none_url(self):
        url = None
        query_params = {"param1": "value1", "param2": "value2"}
        expected_url = None
        output_url = add_query_param_in_url(url, query_params)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={query_params}) but got `{output_url}"


class TestConcatenateParamInUrl:
    @pytest.mark.unit
    def test_add_valid_two_param_with_sep(self):
        url = "http://www.example.com"
        params = ["value1", "value2"]
        sep = "-"
        expected_url = "http://www.example.com-value1-value2"
        output_url = concatenate_params_in_url(url, params, sep)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={params}, sep={sep}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_valid_two_param_with_sep_none_value(self):
        url = "http://www.example.com"
        params = ["value1", None]
        sep = "-"
        expected_url = "http://www.example.com-value1"
        output_url = concatenate_params_in_url(url, params, sep)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={params}, sep={sep}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_empty_params_with_valid_sep(self):
        url = "http://www.example.com"
        params = []
        sep = "-"
        expected_url = "http://www.example.com"
        output_url = concatenate_params_in_url(url, params, sep)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={params}, sep={sep}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_none_param(self):
        url = "http://www.example.com"
        params = None
        sep = "-"
        expected_url = "http://www.example.com"
        output_url = concatenate_params_in_url(url, params, sep)
        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={params}, sep={sep}) but got `{output_url}"

    @pytest.mark.unit
    def test_add_param_in_none_url(self):
        url = None
        params = ["value1", "value2"]
        sep = "-"
        expected_url = None
        output_url = concatenate_params_in_url(url, params, sep)

        assert (
            output_url == expected_url
        ), f"Expected `{expected_url}` for (url={url}, query={params}, sep={sep}) but got `{output_url}"


class TestReplaceSep:
    @pytest.mark.unit
    def test_valid_string_with_one_sep(self):
        x = "test|test1"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_with_final_char_is_sep(self):
        x = "test|test1|"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_without_sep(self):
        x = "test test1"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test test1"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_with_only_new_sep(self):
        x = "test,test1"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_valid_string_with_final_new_sep(self):
        x = "test.test1."
        current_sep = "|"
        new_sep = "."
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test.test1"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_empty_string(self):
        x = ""
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = ""

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_multiple_sep(self):
        x = "test|test1|test2|test3"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1,test2,test3"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_multiple_new_sep(self):
        x = "test,test1,test2,test3"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1,test2,test3"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_both_sep(self):
        x = "test,test1|test2,test3"
        current_sep = "|"
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1,test2,test3"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}`"

    @pytest.mark.unit
    def test_string_with_whitespace_sep(self):
        x = "test test1"
        current_sep = " "
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = "test,test1"

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}"

    @pytest.mark.unit
    def test_with_none_input(self):
        x = None
        current_sep = " "
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = None

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}"

    @pytest.mark.unit
    def test_with_not_string_input(self):
        x = 3
        current_sep = " "
        new_sep = ","
        output = replace_sep(x, current_sep, new_sep)
        expected_output = 3

        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({x}, {current_sep}, {new_sep}) but got `{output}"


class TestClosestValue:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("input_value", "expected_value"),
        [[0.5, 0.6], [0, 0.2], [0.522, 0.6], [1.15, 1], [0.8, 0.8]],
    )
    def test_above_value(self, input_value, expected_value):
        v = closest_above_value([0.2, 0.4, 0.6, 0.8, 1], input_value)

        assert v == expected_value

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("input_value", "expected_value"),
        [[0.5, 0.4], [0, 0.2], [0.522, 0.4], [1.15, 1], [0.8, 0.8]],
    )
    def test_below_value(self, input_value, expected_value):
        v = closest_below_value([0.2, 0.4, 0.6, 0.8, 1], input_value)

        assert v == expected_value


class TestIterateAll:
    @pytest.mark.unit
    def test_iterate_all_should_return_an_generator(self):
        assert isinstance(iterate_all({"1": 1}), Generator)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("expected", "args", "assert_desc"),
        [
            ([1], {"iterable": {"1": 1}}, "should return [1] for dict {'1': 1}"),
            ([2], {"iterable": {"2": 2}}, "should return [2] for dict {'2': 2}"),
            ([3], {"iterable": {"3": 3}}, "should return [3] for dict {'3': 3}"),
            (
                [1, 2],
                {"iterable": {"1": 1, "2": 2}},
                "should return [1, 2] for dict {'1': 1, '2': 2}",
            ),
            (
                [1, 2, 3],
                {"iterable": {"1": 1, "2": 2, "3": 3}},
                "should return [1, 2, 3] for dict {'1': 1, '2': 2, '3': 3}",
            ),
            (
                [1],
                {"iterable": {"1": {"1": 1}}},
                'should return [1], for dict {"1": {"1": 1}}}',
            ),
            (
                [1, 2],
                {"iterable": {"1": {"1": 1}, "2": 2}},
                'should return [1, 2], for dict {"1": {"1": 1}, "2": 2}}',
            ),
            (
                [1, 2],
                {"iterable": {"1": {"1": 1, "2": 2}}},
                'should return [1, 2], for dict {"1": {"1": 1, "2": 2}}}',
            ),
            (
                [1],
                {"iterable": [1]},
                "should return [1], for list [1]",
            ),
            (
                [2],
                {"iterable": [2]},
                "should return [2], for list [2]",
            ),
            (
                [1, 2],
                {"iterable": [1, 2]},
                "should return [1, 2], for list [1, 2]",
            ),
            (
                [1, 2, 3],
                {"iterable": [1, 2, 3]},
                "should return [1, 2, 3], for list [1, 2, 3]",
            ),
            (
                [1, 2, 3],
                {"iterable": [1, [2, 3]]},
                "should return [1, 2, 3], for list [1, [2, 3]]",
            ),
            (
                [1, 2, 3],
                {"iterable": {"1": 1, "2": [2, 3]}},
                'should return [1, 2, 3], for dict {"1": 1, "2": [2, 3]}',
            ),
            (
                [1, 2, 3],
                {"iterable": [1, {"2": [2, 3]}]},
                'should return [1, 2, 3], for dict [1, {"2": [2, 3]}]}',
            ),
            (
                [1, 2, 3],
                {"iterable": {"1": 1, "2": [{"2": 2}, {"3": 3}]}},
                'should return [1, 2, 3], for dict {"1": 1, "2": [{"2": 2}, {"3": 3}]}',
            ),
            (
                ["1"],
                {"iterable": {"1": 1}, "returned": "key"},
                "should return ['1'] for dict {'1': 1}",
            ),
            (
                ["1", "2"],
                {"iterable": {"1": 1, "2": [2, 3]}, "returned": "key"},
                'should return [1, 2, 3], for dict {"1": 1, "2": [{"2": 2}, {"3": 3}]}',
            ),
        ],
    )
    def test_basicTest(self, expected, args, assert_desc):
        output = [it for it in iterate_all(**args)]
        assert expected == output, assert_desc

    @pytest.mark.unit
    def test_bad_value_for_returned(self):
        with pytest.raises(ValueError):
            ret = [it for it in iterate_all(iterable={"1": 1}, returned="Bad")]
            assert [1] == ret
