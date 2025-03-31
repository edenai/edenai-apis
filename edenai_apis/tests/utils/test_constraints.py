import mimetypes
import os
from typing import Optional

import pytest
from pytest_mock import MockerFixture
from settings import base_path

from edenai_apis.utils import constraints
from edenai_apis.utils.constraints import (
    validate_all_input_languages,
    validate_input_file_type,
    validate_single_language,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.files import FileInfo, FileWrapper
from edenai_apis.utils.languages import LanguageErrorMessage

PROVIDER = "test_provider"
FEATURE = "test_feature"
SUBFEATURE = "test_subfeature"


class TestValidateInputFileType:
    @staticmethod
    def failure_message(expected_output, output, **kwargs):
        """Return a format message for test fail

        Args:
            expected_output(Any): _description_
            output(Any): _description_
            **kwargs: All inputs parameters of the function called
        """
        return f"Expected `{expected_output}` for parameters `{kwargs}`, but got `{output}`"

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "constraints",
        [{"file_types": ["image/png", "image/jpeg"]}, {"file_types": ["image/*"]}, {}],
        ids=[
            "test_with_specific_file_types",
            "test_with_generic_file_type",
            "test_without_file_type_constraints",
        ],
    )
    def test_valid_constraints_and_file(self, constraints):
        provider = "faker"
        file_path = os.path.join(base_path, "features/image/data/objects.png")
        mime_type = mimetypes.guess_type(file_path)[0]
        file_info = FileInfo(
            os.stat(file_path).st_size,
            mime_type,
            [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
            "",
            "",
        )
        file_wrapper = FileWrapper(file_path, "", file_info)
        args = {"file": file_wrapper}

        output = validate_input_file_type(
            constraints=constraints, provider=provider, args=args
        )

        assert output == args, TestValidateInputFileType.failure_message(
            expected_output=args,
            output=output,
            constraints=constraints,
        )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "file",
        [
            os.path.join(base_path, "features/image/data/explicit_content.jpeg"),
            os.path.join(base_path, "features/image/data/face_recognition_1.jpg"),
            os.path.join(base_path, "features/audio/data/out.wav"),
        ],
        ids=["test_with_jpeg", "test_with_jpg", "test_with_wav"],
    )
    def test_invalid_file(self, file):
        # Setup
        provider = "faker"
        constraints = {"file_types": ["image/png"]}
        mime_type = mimetypes.guess_type(file)[0]
        file_info = FileInfo(
            os.stat(file).st_size,
            mime_type,
            [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
            "",
            "",
        )
        file_wrapper = FileWrapper(file, "", file_info)
        args = {"file": file_wrapper}

        # Action
        with pytest.raises(
            ProviderException, match=f"Provider {provider} doesn't support file type:"
        ):
            validate_input_file_type(
                constraints=constraints, provider=provider, args=args
            )


class TestValidateSingleLanguage:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("language", "expected_output", "null_language", "ret_mock_function"),
        [
            [{"key": "language", "value": "en"}, "en", False, "en"],
            [
                {"key": "target_language", "value": "en-US"},
                "en",
                False,
                "en",
            ],
            [
                {"key": "language", "value": None},
                None,
                True,
                None,
            ],
            [
                {"key": "source_language", "value": "auto-detect"},
                None,
                True,
                None,
            ],
            [{"key": "language", "value": "AUTO-DETECT"}, None, True, None],
        ],
        ids=[
            "test_with_simple_langue",
            "test_with_langue_and_territory",
            "test_without_langue_and_null_language_true",
            "test_with_auto-detect_language",
            "test_with_AUTO-DETECT_language",
        ],
    )
    def test_valid_language(
        self,
        mocker: MockerFixture,
        language: str,
        expected_output: Optional[str],
        null_language: bool,
        ret_mock_function: Optional[str],
    ):
        # Create mock for provide_appropriate_language
        mocker.patch(
            "edenai_apis.utils.constraints.provide_appropriate_language",
            return_value=ret_mock_function,
        )

        # Action
        output = validate_single_language(
            provider_name=PROVIDER,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            language=language,
            null_language_accepted=null_language,
        )

        assert output == expected_output

    @pytest.mark.unit
    @pytest.mark.parametrize(
        (
            "language",
            "ret_mock_load",
            "expected_raise",
            "ret_mock_provide",
        ),
        [
            [
                {"key": "language", "value": None},
                None,
                LanguageErrorMessage.LANGUAGE_REQUIRED("language"),
                None,
            ],
            [
                {"key": "language", "value": "abc"},
                [],
                LanguageErrorMessage.LANGUAGE_NOT_SUPPORTED("abc", "language"),
                None,
            ],
            [
                {"key": "language", "value": "fr-FR"},
                ["fr"],
                LanguageErrorMessage.LANGUAGE_GENERIQUE_REQUESTED(
                    "fr-FR", "fr", "language"
                ),
                None,
            ],
            [
                {"key": "language", "value": "fr_FR_123"},
                ["fr"],
                LanguageErrorMessage.LANGUAGE_SYNTAX_ERROR("fr_FR_123"),
                SyntaxError,
            ],
        ],
        ids=[
            "test_null_language_not_accepted",
            "test_unsupported_language",
            "test_too_much_specific_language",
            "test_syntax_error",
        ],
    )
    def test_invalid_language(
        self,
        mocker: MockerFixture,
        language: dict,
        ret_mock_load: list,
        expected_raise: str,
        ret_mock_provide,
    ):
        # Create mock for provide_appropriate_language and load_standardize_language
        def fake_provider_appropriate_language(*args, **kwargs):
            if ret_mock_provide:
                raise ret_mock_provide
            return None

        mocker.patch(
            "edenai_apis.utils.constraints.provide_appropriate_language",
            side_effect=fake_provider_appropriate_language,
        )

        mocker.patch(
            "edenai_apis.utils.constraints.load_standardized_language",
            return_value=ret_mock_load,
        )

        # Try to except ProviderException with specific error message
        with pytest.raises(ProviderException) as excinfo:
            validate_single_language(
                provider_name=PROVIDER,
                feature=FEATURE,
                subfeature=SUBFEATURE,
                language=language,
                null_language_accepted=False,
            )
        assert str(excinfo.value) == expected_raise


class TestValidateAllInputLanguages:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("args", "number_of_call"),
        [
            [{"language": "en"}, 1],
            [{"language": "en", "test": "Do not a language"}, 1],
            [{"source_language": "en", "target_language": "fr"}, 2],
            [{"test": "Do not a language"}, 0],
        ],
        ids=[
            "test_args_with_language",
            "test_args_with_language_and_other",
            "test_args_with_src_and_target_language",
            "test_args_without_language",
        ],
    )
    def test_nb_of_call_validate_single_language(
        self, mocker: MockerFixture, args, number_of_call
    ):
        # Create mock for validate_single_language
        mocker.patch(
            "edenai_apis.utils.constraints.validate_single_language", return_value=None
        )

        validate_all_input_languages(
            constraints={"allow_null_language": False},
            args=args,
            provider_name=PROVIDER,
            feature=FEATURE,
            subfeature=SUBFEATURE,
        )
        assert constraints.validate_single_language.call_count == number_of_call

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("args", "ret_mock", "expected_output"),
        [
            [{"language": "en-US"}, "en", {"language": "en"}],
            [
                {"language": "en", "test": "Do not a language"},
                "fr",
                {"language": "fr", "test": "Do not a language"},
            ],
            [
                {"source_language": "en-US", "target_language": "en"},
                "en-US",
                {"source_language": "en-US", "target_language": "en-US"},
            ],
            [{"test": "Do not a language"}, None, {"test": "Do not a language"}],
        ],
        ids=[
            "test_args_with_language",
            "test_args_with_language_and_other",
            "test_args_with_src_and_target_language",
            "test_args_without_language",
        ],
    )
    def test_default_function(
        self, mocker: MockerFixture, args, ret_mock, expected_output
    ):
        # Create mock for validate_single_language
        mocker.patch(
            "edenai_apis.utils.constraints.validate_single_language",
            return_value=ret_mock,
        )

        output = validate_all_input_languages(
            constraints={"allow_null_language": False},
            args=args,
            provider_name=PROVIDER,
            feature=FEATURE,
            subfeature=SUBFEATURE,
        )
        assert output == expected_output
