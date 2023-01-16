"""
    Test language standardization functions.
    Different Providers handle different language formats.
    We implement language utils to handle the standardisation.
"""
import pytest
from edenai_apis.utils.languages import (
    check_language_format,
    convert_three_two_letters,
    load_language_constraints
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

    def test_feature_without_languages_in_constraints(self):
        assert load_language_constraints('mindee', 'ocr', 'identity_parser') == [], \
            "mindee-ocr-identity_parser have no language constraints"

    def test_feature_without_constraints(self):
        assert load_language_constraints('google', 'image', 'face_detection') == [], \
            "google-image-face_detection have no constraints"
