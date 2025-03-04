"""
Test language standardization functions.
Different Providers handle different language formats.
We implement language utils to handle the standardisation.
"""

import pytest
from pytest_mock import MockerFixture

from edenai_apis.utils.languages import (
    AUTO_DETECT,
    AUTO_DETECT_NAME,
    check_language_format,
    compare_language_and_region_code,
    convert_three_two_letters,
    expand_languages_for_user,
    format_language_name,
    get_code_from_language_name,
    get_language_name_from_code,
    load_language_constraints,
    load_standardized_language,
    provide_appropriate_language,
)


class TestCheckLanguageFormat:
    @pytest.mark.unit
    def test_valid_language_code(self):
        assert check_language_format("en") == True, '"en" should be a valid language'

    @pytest.mark.unit
    def test_valid_language_code_with_region(self):
        assert (
            check_language_format("en-US") == True
        ), '"en-US" should be a valid language'

    @pytest.mark.unit
    def test__valid_language_code_with_script_and_region(self):
        assert (
            check_language_format("en-Latn-US") == True
        ), '"en-Latn-US" should be a valid language'

    @pytest.mark.unit
    def test_invalid_language_code(self):
        assert (
            check_language_format("abcd") == False
        ), '"abcd" should be an invalid language'

    @pytest.mark.unit
    def test_invalid_language_code_with_only_number(self):
        assert (
            check_language_format("1234") == False
        ), '"1234" should be an invalid language'

    @pytest.mark.unit
    def test__invalid_language_code_with_region(self):
        assert (
            check_language_format("en-") == False
        ), '"en-" should be an invalid language'

    @pytest.mark.unit
    def test_invalid_language_code_with_script_and_region(self):
        assert (
            check_language_format("en-Latn-") == False
        ), '"en-Latn-" should be an invalid language'
        assert (
            check_language_format("en-Latn-US-") == False
        ), '"en-Latn-US-" should be an invalid language'
        assert (
            check_language_format("en-Latn-US-123") == False
        ), '"en-Latn-US-123" should be an invalid language'

    @pytest.mark.unit
    def test_none_input(self):
        assert (
            check_language_format(None) == None
        ), "None should not be a language format"


class TestConvertThreeTwoLetters:
    @pytest.mark.unit
    def test_valid_iso639_3_code(self):
        assert (
            convert_three_two_letters("fra") == "fr"
        ), "The iso639_2 for fra must be fr"

    @pytest.mark.unit
    def test_valid_iso639_3_code_with_region(self):
        assert (
            convert_three_two_letters("fra-FR") == "fr-FR"
        ), "The iso639_2 for fra-FR must be fr-FR"

    @pytest.mark.unit
    def test_valid_iso639_2_code(self):
        assert convert_three_two_letters("fr") == "fr", "The iso639_2 for fr must be fr"

    @pytest.mark.unit
    def test_invalid_code(self):
        assert (
            convert_three_two_letters("abcd") == "abcd"
        ), "The iso639_2 for abcd must be abcd"

    @pytest.mark.unit
    def test_none_input(self):
        assert (
            convert_three_two_letters(None) == None
        ), "The iso639_2 for None must be None"


class TestLoadLanguageConstraints:
    PROVIDER = "test_provider"
    FEATUTRE = "test_feature"
    SUBFEATURE = "test_subfeature"

    @pytest.mark.unit
    def test_valid_provider_feature_and_subfeature(self, mocker: MockerFixture):
        ret_mock_value = {"constraints": {"languages": ["en", "fr"]}}
        mocker.patch(
            "edenai_apis.utils.languages.load_provider", return_value=ret_mock_value
        )
        output = load_language_constraints(
            self.PROVIDER, self.FEATUTRE, self.SUBFEATURE
        )
        expected_output = ["en", "fr"]
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_provider_allow_null_language(self, mocker: MockerFixture):
        ret_mock_value = {
            "constraints": {
                "languages": ["en", "fr"],
                "allow_null_language": True,
            }
        }
        mocker.patch(
            "edenai_apis.utils.languages.load_provider", return_value=ret_mock_value
        )
        output = load_language_constraints(
            self.PROVIDER, self.FEATUTRE, self.SUBFEATURE
        )
        expected_output = ["en", "fr", "auto-detect"]
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` but got `{output}`"

    @pytest.mark.unit
    def test_invalid_provider_allow_null_language(self, mocker: MockerFixture):
        ret_mock_value = {
            "constraints": {
                "languages": ["en", "fr"],
                "allow_null_language": False,
            }
        }
        mocker.patch(
            "edenai_apis.utils.languages.load_provider", return_value=ret_mock_value
        )
        output = load_language_constraints(
            self.PROVIDER, self.FEATUTRE, self.SUBFEATURE
        )
        expected_output = ["en", "fr"]
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` but got `{output}`"

    @pytest.mark.unit
    def test_feature_without_languages_in_constraints(self, mocker: MockerFixture):
        ret_mock_value = {
            "constraints": {
                "language": ["en", "fr"],
                "allow_null_language": False,
            }
        }
        mocker.patch(
            "edenai_apis.utils.languages.load_provider", return_value=ret_mock_value
        )
        output = load_language_constraints(
            self.PROVIDER, self.FEATUTRE, self.SUBFEATURE
        )
        expected_output = []
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` but got `{output}`"

    @pytest.mark.unit
    def test_feature_without_constraints(self, mocker: MockerFixture):
        ret_mock_value = {
            "constraint": {
                "language": ["en", "fr"],
                "allow_null_language": False,
            }
        }
        mocker.patch(
            "edenai_apis.utils.languages.load_provider", return_value=ret_mock_value
        )
        output = load_language_constraints(
            self.PROVIDER, self.FEATUTRE, self.SUBFEATURE
        )
        expected_output = []
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` but got `{output}`"


class TestExpandLanguagesForUser:
    @pytest.mark.unit
    def test_valid_list_languages(self):
        result = expand_languages_for_user(["auto-detect", "en", "fra", "it-IT"])
        assert result == [
            "auto-detect",
            "en",
            "fr",
            "it",
            "it-IT",
        ], "['auto-detect', 'en', 'fra', 'it-IT'] must be convert to ['auto-detect', 'en', 'fra', 'it', 'it-IT']"

    @pytest.mark.unit
    def test_valid_list_languages_with_bad_language(self):
        result = expand_languages_for_user(["en", "it-IT", "abc"])
        assert result == [
            "en",
            "it",
            "it-IT",
            "abc",
        ], "['en', 'it-IT', 'abc'] must be convert to ['en', 'it', 'it-IT', 'abc']"

    @pytest.mark.unit
    def test_list_language_empty(self):
        assert (
            expand_languages_for_user([]) == []
        ), "A empty list of language must be stay empty"

    @pytest.mark.unit
    def test_list_language_none(self):
        with pytest.raises(TypeError):
            expand_languages_for_user(None)


class TestLoadStandardizedLanguage:
    @pytest.mark.unit
    def test_valid_input_with_multi_provider(self, mocker: MockerFixture):
        mocker.patch(
            "edenai_apis.utils.languages.load_language_constraints",
            side_effect=[["en", "fr"], ["fr", "es"]],
        )
        providers = ["provider1", "provider2"]
        feature = "test_feature"
        subfeature = "test_subfeature"
        expected_output = ["en", "fr", "es"]

        output = load_standardized_language(feature, subfeature, providers)
        assert sorted(output) == sorted(
            expected_output
        ), f"Expected `{expected_output}` but got `{output}`"


class TestFormatLanguageName:
    @pytest.mark.unit
    def test_unknown_language(self):
        language_name = "Unknown language (US)"
        isocode = "US"
        expected_output = "Region: US"
        output = format_language_name(language_name, isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    @pytest.mark.unit
    def test_unknown_region(self):
        language_name = "English (Unknown Region)"
        isocode = "en"
        expected_output = "English"
        output = format_language_name(language_name, isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    @pytest.mark.unit
    def test_unknown_region_and_language(self):
        language_name = "Unknown language (Unknown Region)"
        isocode = "abc-abc"
        expected_output = "Region: Unknown Region"
        output = format_language_name(language_name, isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    @pytest.mark.unit
    def test_valid_language(self):
        language_name = "English"
        isocode = "en"
        expected_output = "English"
        output = format_language_name(language_name, isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    @pytest.mark.unit
    def test_valid_language_and_region(self):
        language_name = "English (US)"
        isocode = "en-US"
        expected_output = "English (US)"
        output = format_language_name(language_name, isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"


class TestGetLanguageNameFromCode:
    @pytest.mark.unit
    def test_none_isocode(self):
        isocode = None
        expected_output = ""
        output = get_language_name_from_code(isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    @pytest.mark.unit
    def test_auto_detect_isocode(self):
        isocode = AUTO_DETECT
        expected_output = AUTO_DETECT_NAME
        output = get_language_name_from_code(isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    @pytest.mark.unit
    def test_two_letter_isocode(self):
        isocode = "en"
        expected_output = "English"
        output = get_language_name_from_code(isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    @pytest.mark.unit
    def test_three_letter_isocode(self):
        isocode = "eng"
        expected_output = "English"
        output = get_language_name_from_code(isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    @pytest.mark.unit
    def test_isocode_with_region(self):
        isocode = "en-US"
        expected_output = "English (United States)"
        output = get_language_name_from_code(isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    @pytest.mark.unit
    def test_invalid_isocode(self):
        isocode = "unknown"
        expected_output = ""
        output = get_language_name_from_code(isocode)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{isocode}` but got `{output}`"


class TestGetCodeFromLanguageName:
    @pytest.mark.unit
    def test_valid_name(self):
        name = "English"
        expected_output = "en"
        output = get_code_from_language_name(name)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{name}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_name_with_region(self):
        name = "English (United States)"
        expected_output = "en"
        output = get_code_from_language_name(name)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{name}` but got `{output}`"

    @pytest.mark.unit
    def test_invalid_name(self):
        name = "InvalidName"
        expected_output = "Unknow"
        output = get_code_from_language_name(name)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{name}` but got `{output}`"

    @pytest.mark.unit
    def test_none_name(self):
        name = None
        expected_output = "Unknow"
        output = get_code_from_language_name(name)
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{name}` but got `{output}`"


class TestCompareLanguageAndRegionCode:
    @pytest.mark.unit
    def test_same_language_and_region(self):
        iso_code = "en-US"
        selected_code_language = "en-US"
        assert (
            compare_language_and_region_code(iso_code, selected_code_language) == True
        )

    @pytest.mark.unit
    def test_same_language_different_region(self):
        iso_code = "en-US"
        selected_code_language = "en-GB"
        assert (
            compare_language_and_region_code(iso_code, selected_code_language) == False
        )

    @pytest.mark.unit
    def test_different_language_same_region(self):
        iso_code = "en-US"
        selected_code_language = "fr-US"
        assert (
            compare_language_and_region_code(iso_code, selected_code_language) == False
        )

    @pytest.mark.unit
    def test_different_language_and_region(self):
        iso_code = "en-US"
        selected_code_language = "fr-FR"
        assert (
            compare_language_and_region_code(iso_code, selected_code_language) == False
        )


class TestProvideAppropriateLanguage:
    PROVIDER = "test_provider"
    FEATURE = "test_feature"
    SUBFEATURE = "test_subfeature"

    @pytest.mark.unit
    def test_valid_input(self, mocker: MockerFixture):
        # Create mock for load_languages_constraints
        return_mock = ["en-US", "fr", "es"]
        mocker.patch(
            "edenai_apis.utils.languages.load_language_constraints",
            return_value=return_mock,
        )

        # Setup
        iso_code = "en"
        expected_output = "en-US"

        # Action
        output = provide_appropriate_language(
            iso_code, self.PROVIDER, self.FEATURE, self.SUBFEATURE
        )

        # Assert
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_input_with_valid_region(self, mocker: MockerFixture):
        # Create mock for load_languages_constraints
        return_mock = ["en-US", "fr", "es"]
        mocker.patch(
            "edenai_apis.utils.languages.load_language_constraints",
            return_value=return_mock,
        )

        # Setup
        iso_code = "en-US"
        expected_output = "en-US"

        # Action
        output = provide_appropriate_language(
            iso_code, self.PROVIDER, self.FEATURE, self.SUBFEATURE
        )

        # Assert
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    @pytest.mark.unit
    def test_valid_input_with_invalid_region(self, mocker: MockerFixture):
        # Create mock for load_languages_constraints
        return_mock = ["en-US", "fr", "es"]
        mocker.patch(
            "edenai_apis.utils.languages.load_language_constraints",
            return_value=return_mock,
        )

        # Setup
        iso_code = "en-EN"
        expected_output = None

        # Action
        output = provide_appropriate_language(
            iso_code, self.PROVIDER, self.FEATURE, self.SUBFEATURE
        )

        # Assert
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    @pytest.mark.unit
    def test_invalid_isocode(self, mocker: MockerFixture):
        # Create mock for load_languages_constraints
        return_mock = ["en-US", "fr", "es"]
        mocker.patch(
            "edenai_apis.utils.languages.load_language_constraints",
            return_value=return_mock,
        )

        # Setup
        iso_code = "inv"
        expected_output = None

        # Action
        output = provide_appropriate_language(
            iso_code, self.PROVIDER, self.FEATURE, self.SUBFEATURE
        )

        # Assert
        assert (
            output == expected_output
        ), f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    @pytest.mark.unit
    def test_invalid_input(self):
        with pytest.raises(SyntaxError):
            iso_code = "12345"
            provide_appropriate_language(
                iso_code, self.PROVIDER, self.FEATURE, self.SUBFEATURE
            )
