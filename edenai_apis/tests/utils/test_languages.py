"""
    Test language standardization functions.
    Different Providers handle different language formats.
    We implement language utils to handle the standardisation.
"""
import pytest
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
    provide_appropriate_language
)

class TestCheckLanguageFormat:
    def test_valid_language_code(self):
        assert check_language_format("en") == True, \
            '"en" should be a valid language'

    def test_valid_language_code_with_region(self):
        assert check_language_format("en-US") == True, \
            '"en-US" should be a valid language'

    def test__valid_language_code_with_script_and_region(self):
        assert check_language_format("en-Latn-US") == True, \
            '"en-Latn-US" should be a valid language'

    def test_invalid_language_code(self):
        assert check_language_format("abcd") == False, \
            '"abcd" should be an invalid language'

    def test_invalid_language_code_with_only_number(self):
        assert check_language_format("1234") == False, \
            '"1234" should be an invalid language'

    def test__invalid_language_code_with_region(self):
        assert check_language_format("en-") == False, \
            '"en-" should be an invalid language'

    def test_invalid_language_code_with_script_and_region(self):
        assert check_language_format("en-Latn-") == False, \
            '"en-Latn-" should be an invalid language'
        assert check_language_format("en-Latn-US-") == False, \
            '"en-Latn-US-" should be an invalid language'
        assert check_language_format("en-Latn-US-123") == False, \
            '"en-Latn-US-123" should be an invalid language'

    def test_none_input(self):
        assert check_language_format(None) == None,  \
            'None should not be a language format'

class TestConvertThreeTwoLetters:
    def test_valid_iso639_3_code(self):
        assert convert_three_two_letters("fra") == "fr", \
            "The iso639_2 for fra must be fr"

    def test_valid_iso639_3_code_with_region(self):
        assert convert_three_two_letters("fra-FR") == "fr-FR", \
           "The iso639_2 for fra-FR must be fr-FR"

    def test_valid_iso639_2_code(self):
        assert convert_three_two_letters("fr") == "fr", \
            "The iso639_2 for fr must be fr"

    def test_invalid_code(self):
        assert convert_three_two_letters("abcd") == "abcd", \
            "The iso639_2 for abcd must be abcd"

    def test_none_input(self):
        assert convert_three_two_letters(None) == None, \
            "The iso639_2 for None must be None"

class TestLoadLanguageConstraints:
    def test_valid_provider_feature_and_subfeature(self):
        result = load_language_constraints('emvista', 'text', 'summarize')
        assert result == ['en', 'fr'], \
            f"emvista-text-summarize must handle 'en' and 'fr' not {result}"

    def test_valid_provider_allow_null_language(self):
        result = load_language_constraints('google', 'ocr', 'ocr')
        assert 'auto-detect' in result, \
            f"google-ocr-ocr must handle the auto detect language"

    def test_invalid_provider_allow_null_language(self):
        result = load_language_constraints('amazon', 'ocr', 'invoice_parser')
        assert 'auto-detect' not in result, \
            f"amazon-ocr-invoice_parser doesn't handle the auto detect language"

    def test_invalid_provider_name(self):
        with pytest.raises(KeyError):
            load_language_constraints('emvistas', 'text', 'summarize')

    def test_invalid_feature(self):
        with pytest.raises(KeyError):
            load_language_constraints('emvista', 'texts', 'summarize')

    def test_invalid_subfeature(self):
        with pytest.raises(KeyError):
            load_language_constraints('emvista', 'text', 'summarizes')

    def test_none_provider_name(self):
        with pytest.raises(KeyError):
            load_language_constraints(None, 'text', 'summarize')

    def test_none_feature(self):
        with pytest.raises(KeyError):
            load_language_constraints('emvista', None, 'summarize')

    def test_none_subfeature(self):
        with pytest.raises(KeyError):
            load_language_constraints('emvista', 'text', None)

    def test_feature_without_languages_in_constraints(self):
        assert load_language_constraints('mindee', 'ocr', 'identity_parser') == [], \
            "mindee-ocr-identity_parser have no language constraints"

    def test_feature_without_constraints(self):
        assert load_language_constraints('google', 'image', 'face_detection') == [], \
            "google-image-face_detection have no constraints"


class TestExpandLanguagesForUser:
    def test_valid_list_languages(self):
        result = expand_languages_for_user(['auto-detect', 'en', 'fra', 'it-IT'])
        assert result == ['auto-detect', 'en', 'fr', 'it', 'it-IT'], \
            "['auto-detect', 'en', 'fra', 'it-IT'] must be convert to ['auto-detect', 'en', 'fra', 'it', 'it-IT']"

    def test_valid_list_languages_with_bad_language(self):
        result = expand_languages_for_user(['en', 'it-IT', 'abc'])
        assert result == ['en', 'it', 'it-IT', 'abc'], \
            "['en', 'it-IT', 'abc'] must be convert to ['en', 'it', 'it-IT', 'abc']"

    def test_list_language_empty(self):
        assert expand_languages_for_user([]) == [], \
            "A empty list of language must be stay empty"

    def test_list_language_none(self):
        with pytest.raises(TypeError):
            expand_languages_for_user(None)


class TestLoadStandardizedLanguage:
    def test_valid_input_with_two_providers(self):
        result = load_standardized_language('text', 'summarize', ['oneai', 'emvista'])
        assert sorted(result) == sorted(['en', 'fr']), \
            "oneai and emvista text.summarize have ['fr', 'en'] in language list"

    def test_valid_input_with_none_providers(self):
        result = load_standardized_language('ocr', 'identity_parser', None)
        expected = [
            'it-IT',
            'fr',
            'de-DE',
            'es-ES',
            'fr-FR',
            'it',
            'pt-PT',
            'en-US',
            'en',
            'es',
            'pt',
            'de'
        ]
        assert sorted(result) == sorted(expected), \
            "Language list for ocr.identity_parser is not accurate"

    def test_valid_input_with_feature_without_language(self):
        assert load_standardized_language('image', 'face_detection', ['google']) == [], \
            "google.image.face_detection has no language constraint, function must be return an"

    def test_invalid_feature(self):
        with pytest.raises(KeyError):
            load_standardized_language('texts', 'summarize', ['oneai'])

    def test_invalid_subfeature(self):
        with pytest.raises(KeyError):
            load_standardized_language('text', 'summarizes', ['oneai'])

    def test_none_feature(self):
        with pytest.raises(KeyError):
            load_standardized_language(None, 'summarize', ['oneai'])

    def test_none_subfeature(self):
        with pytest.raises(KeyError):
            load_standardized_language('text', None, ['oneai'])

class TestFormatLanguageName:
    def test_unknown_language(self):
        language_name = "Unknown language (US)"
        isocode = "US"
        expected_output = "Region: US"
        output = format_language_name(language_name, isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    def test_unknown_region(self):
        language_name = "English (Unknown Region)"
        isocode = "en"
        expected_output = "English"
        output = format_language_name(language_name, isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    def test_unknown_region_and_language(self):
        language_name = "Unknown language (Unknown Region)"
        isocode = "abc-abc"
        expected_output = "Region: Unknown Region"
        output = format_language_name(language_name, isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    def test_valid_language(self):
        language_name = "English"
        isocode = "en"
        expected_output = "English"
        output = format_language_name(language_name, isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

    def test_valid_language_and_region(self):
        language_name = "English (US)"
        isocode = "en-US"
        expected_output = "English (US)"
        output = format_language_name(language_name, isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for ({language_name}, {isocode}) but got `{output}`"

class TestGetLanguageNameFromCode:
    def test_none_isocode(self):
        isocode = None
        expected_output = ""
        output = get_language_name_from_code(isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    def test_auto_detect_isocode(self):
        isocode = AUTO_DETECT
        expected_output = AUTO_DETECT_NAME
        output = get_language_name_from_code(isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    def test_two_letter_isocode(self):
        isocode = "en"
        expected_output = "English"
        output = get_language_name_from_code(isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    def test_three_letter_isocode(self):
        isocode = "eng"
        expected_output = "English"
        output = get_language_name_from_code(isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

    def test_isocode_with_region(self):
        isocode = "en-US"
        expected_output = "English (United States)"
        output = get_language_name_from_code(isocode)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{isocode}` but got `{output}`"

class TestGetCodeFromLanguageName:
    def test_valid_name(self):
        name = "English"
        expected_output = "en"
        output = get_code_from_language_name(name)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{name}` but got `{output}`"

    def test_valid_name_with_region(self):
        name = "English (United States)"
        expected_output = "en"
        output = get_code_from_language_name(name)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{name}` but got `{output}`"

    def test_invalid_name(self):
        name = "InvalidName"
        expected_output = "Unknow"
        output = get_code_from_language_name(name)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{name}` but got `{output}`"

    def test_none_name(self):
        name = None
        expected_output = "Unknow"
        output = get_code_from_language_name(name)
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{name}` but got `{output}`"

class TestCompareLanguageAndRegionCode:
    def test_same_language_and_region(self):
        iso_code = 'en-US'
        selected_code_language = 'en-US'
        assert compare_language_and_region_code(iso_code, selected_code_language) == True

    def test_same_language_different_region(self):
        iso_code = 'en-US'
        selected_code_language = 'en-GB'
        assert compare_language_and_region_code(iso_code, selected_code_language) == False

    def test_different_language_same_region(self):
        iso_code = 'en-US'
        selected_code_language = 'fr-US'
        assert compare_language_and_region_code(iso_code, selected_code_language) == False

    def test_different_language_and_region(self):
        iso_code = 'en-US'
        selected_code_language = 'fr-FR'
        assert compare_language_and_region_code(iso_code, selected_code_language) == False

class TestProvideAppropriateLanguage:
    def test_valid_input(self):
        iso_code = 'en'
        provider_name = 'google'
        feature = 'ocr'
        subfeature = 'ocr'
        output = provide_appropriate_language(iso_code, provider_name, feature, subfeature)
        expected_output = 'en-US'
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    def test_valid_input_with_valid_region(self):
        iso_code = 'en-US'
        provider_name = 'google'
        feature = 'ocr'
        subfeature = 'ocr'
        output = provide_appropriate_language(iso_code, provider_name, feature, subfeature)
        expected_output = 'en-US'
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    def test_valid_input_with_invalid_region(self):
        iso_code = 'en-EN'
        provider_name = 'google'
        feature = 'ocr'
        subfeature = 'ocr'
        output = provide_appropriate_language(iso_code, provider_name, feature, subfeature)
        expected_output = None
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    def test_invalid_isocode(self):
        iso_code = 'inv'
        provider_name = 'google'
        feature = 'ocr'
        subfeature = 'ocr'
        output = provide_appropriate_language(iso_code, provider_name, feature, subfeature)
        expected_output = None
        assert output == expected_output, \
            f"Expected `{expected_output}` for `{iso_code}` but got `{output}`"

    def test_invalid_input(self):
        with pytest.raises(SyntaxError):
            iso_code = '12345'
            provider_name = 'google'
            feature = 'ocr'
            subfeature = 'ocr'
            provide_appropriate_language(iso_code, provider_name, feature, subfeature)