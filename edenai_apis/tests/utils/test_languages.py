"""
    Test language standardization functions.
    Different Providers handle different language formats.
    We implement language utils to handle the standardisation.
"""
import pytest
from edenai_apis.utils.languages import (
    check_language_format,
    convert_three_two_letters
)

class TestCheckLanguageFormat:
    def test_valid_language_code(self):
        assert check_language_format("en") == True, \
            'check_language_format("en") must be True'

    def test_valid_language_code_with_region(self):
        assert check_language_format("en-US") == True, \
            'check_language_format("en-US") must be True'

    def test__valid_language_code_with_script_and_region(self):
        assert check_language_format("en-Latn-US") == True, \
            'check_language_format("en-Latn-US") must be True'

    def test_invalid_language_code(self):
        assert check_language_format("abcd") == False, \
            'check_language_format("abcd") must be False'

    def test__invalid_language_code_with_region(self):
        assert check_language_format("en-") == False, \
            'check_language_format("en-") must be False'

    def test_invalid_language_code_with_script_and_region(self):
        assert check_language_format("en-Latn-") == False, \
            'check_language_format("en-Latn-") must be False'
        assert check_language_format("en-Latn-US-") == False, \
            'check_language_format("en-Latn-US-") must be False'
        assert check_language_format("en-Latn-US-123") == False, \
            'check_language_format("en-Latn-US-123") must be False'

    def test_none_input(self):
        assert check_language_format(None) == None,  \
            'check_language_format(None) must be None'

class TestConvertThreeTwoLetters:
    def test_valid_iso639_3_code(self):
        assert convert_three_two_letters("fra") == "fr", "valid iso639-3 to iso639-2 conversion test failed"
    def test_valid_iso639_3_code_with_region(self):
        assert convert_three_two_letters("fra-FR") == "fr-FR", "valid iso639-3 with region to iso639-2 conversion test failed"
    def test_valid_iso639_2_code(self):
        assert convert_three_two_letters("fr") == "fr", "valid iso639-2 code test failed"
    def test_invalid_code(self):
        assert convert_three_two_letters("abc") == "abc", "invalid code test failed"
    def test_none_input(self):
        assert convert_three_two_letters(None) == None, "test none input failed"