import pytest
from pytest_mock import MockerFixture

from edenai_apis.utils.constraints import (
    validate_input_file_type,
    validate_single_language
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import LanguageErrorMessage

class TestValidateInputFileType:
    @staticmethod
    def failure_message(expected_output, output, **kwargs):
        """ Return a format message for test fail

        Args:
            expected_output(Any): _description_
            output(Any): _description_
            **kwargs: All inputs parameters of the function called
        """
        return f"Expected `{expected_output}` for parameters `{kwargs}`, but got `{output}`"

    @pytest.mark.parametrize(
        'constraints',
        [
            { 'file_types': ['image/png', 'image/jpeg']},
            { 'file_types': ['image/*']},
            {}
        ]
    )
    def test_valid_constraints_and_file(self, constraints):
        provider = 'faker'
        with open('edenai_apis/features/image/data/objects.png', 'rb') as file:
            args = { 'file': file }

            output = validate_input_file_type(
                constraints=constraints,
                provider=provider,
                args=args
            )

            assert output == args, TestValidateInputFileType.failure_message(
                expected_output=args,
                output=output,
                constraints=constraints,
            )

    @pytest.mark.parametrize('file', [
        open('edenai_apis/features/image/data/explicit_content.jpeg', 'rb'),
        open('edenai_apis/features/image/data/face_recognition_1.jpg', 'rb'),
        open('edenai_apis/features/audio/data/out.wav', 'rb')
    ])
    def test_invalid_file(self, file):
        #Setup
        provider = 'faker'
        constraints = { 'file_types': ['image/png']}
        args = { 'file': file }

        # Action
        with pytest.raises(
            ProviderException,
            match=f"Provider {provider} doesn't support file type:"
        ):
            validate_input_file_type(
                constraints=constraints,
                provider=provider,
                args=args
            )

# TODO add mock for provide_appropriate_language to launch the test
class TestValidateSingleLanguage:
    def test_valid_language(self):
        # Test input with valid language
        provider_name = "test_provider"
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = "en"
        null_language_accepted = False
        input_language = "language"
        expected_output = "en"

        assert validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language) == expected_output

    def test_null_language_not_accepted(self):
        # Test input with None language when provider doesn't auto-detect languages
        provider_name = "test_provider"
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = None
        null_language_accepted = False
        input_language = "language"

        with pytest.raises(ProviderException) as excinfo:
            validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language)
        assert str(excinfo.value) == LanguageErrorMessage.LANGUAGE_REQUIRED("language")

    def test_auto_detect_language(self):
        # Test input with "auto-detect" language
        provider_name = "test_provider"
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = "auto-detect"
        null_language_accepted = True
        input_language = "language"
        expected_output = None

        assert validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language) == expected_output

    def test_unsupported_language(self):
        # Test input with unsupported language
        provider_name = "test_provider"
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = "unsupported_language"
        null_language_accepted = False
        input_language = "language"

        with pytest.raises(ProviderException) as excinfo:
            validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language)
        assert str(excinfo.value) == LanguageErrorMessage.LANGUAGE_NOT_SUPPORTED("unsupported_language", "language")


    def test_generic_language(self):
        # Test input with generic language
        provider_name = "test_provider"
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = "fr-FR"
        null_language_accepted = False
        input_language = "language"

        with pytest.raises(ProviderException) as excinfo:
            validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language)
        assert str(excinfo.value) == LanguageErrorMessage.LANGUAGE_GENERIQUE_REQUESTED("fr-FR", "fr", "language")

    def test_invalid_input(self):
        # Test input with invalid provider name
        provider_name = ""
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = "en"
        null_language_accepted = False
        input_language = "language"

        with pytest.raises(ProviderException) as excinfo:
            validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language)
        assert str(excinfo.value) == "Invalid provider name"

    def test_syntax_error(self):
        # Test input with invalid language format
        provider_name = "test_provider"
        feature = "test_feature"
        subfeature = "test_subfeature"
        language = "en_US"
        null_language_accepted = False
        input_language = "language"

        with pytest.raises(ProviderException) as excinfo:
            validate_single_language(provider_name, feature, subfeature, language, null_language_accepted, input_language)
        assert str(excinfo.value) == "Invalid language format"
