import mimetypes
import os
import re
from time import sleep

import pytest
from apis.amazon.errors import ERRORS as amazon_errors
from apis.google.errors import ERRORS as google_errors
from apis.microsoft.errors import ERRORS as microsoft_errors
from features.audio.speech_to_text_async.speech_to_text_async_args import (
    data_path as audio_data_path,
)
from features.image.anonymization.anonymization_args import data_path as image_data_path
from features.ocr.resume_parser.resume_parser_args import resume_parser_arguments
from pydub.utils import mediainfo

from edenai_apis.interface import compute_output, get_async_job_result
from edenai_apis.loaders.data_loader import FeatureDataEnum
from edenai_apis.loaders.loaders import load_feature
from edenai_apis.utils.exception import (
    ProviderException,
    ProviderInvalidInputAudioDurationError,
    ProviderInvalidInputFileError,
    ProviderInvalidInputFileFormatError,
    ProviderInvalidInputFileSizeError,
    ProviderInvalidInputImageResolutionError,
    ProviderInvalidInputTextLengthError,
    ProviderMissingInputError,
    ProviderParsingError,
)
from edenai_apis.utils.files import FileInfo, FileWrapper


@pytest.mark.e2e
class TestProviderErrors:
    def test_input_text_length_audio_ssml(self):
        error = google_errors[ProviderInvalidInputTextLengthError][0]
        max = 5000
        unit = "bytes"
        input_field = "text"
        feature = "audio"
        subfeature = "text_to_speech"
        text = "This is a too long text. " * 200 + "End."
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        assert len(text.encode("utf-8")) > max, g(
            "test input should be more than provider max limit of "
            f"{max} {unit}, got {len(text.encode('utf-8'))}"
        )
        args[input_field] = text

        with pytest.raises(ProviderException) as exc:
            compute_output("google", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputTextLengthError, (
            "excepiton raised wasn't the right one, expected "
            f"{ProviderInvalidInputTextLengthError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

        # should not raise an error this time
        args[input_field] = text = text[:20]
        assert len(text.encode("utf-8")) < max, (
            "test input should be less than provider max limit of "
            f"{max} {unit}, got {len(text.encode('utf-8'))}"
        )
        response = compute_output("google", feature, subfeature, args)
        assert response is not None

    def test_input_text_too_long_sentences(self):
        error = google_errors[ProviderInvalidInputTextLengthError][0]
        unit = "sentence_count"
        input_field = "text"
        feature = "audio"
        subfeature = "text_to_speech"
        text = "This is a sentence is too long " * 4000 + "."
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        args[input_field] = text

        with pytest.raises(ProviderException) as exc:
            compute_output("google", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputTextLengthError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputTextLengthError.__name__} got {exc.type}"
        )

        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

        # should not raise an error this time
        args[input_field] = text = text[:5] + "."
        response = compute_output("google", feature, subfeature, args)
        assert response is not None

    def test_input_text_too_long(self):
        error = google_errors[ProviderInvalidInputTextLengthError][2]
        max = 102400
        unit = "bytes"
        input_field = "text"
        feature = "translation"
        subfeature = "language_detection"
        text = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent molestie laoreet diam ut ultrices. Nulla convallis purus eget tellus luctus fringilla. Etiam ut libero ut purus consequat sodales. Donec vulputate volutpat mauris, vitae scelerisque nisl blandit nec. Pellentesque euismod vulputate ipsum sit amet dignissim. Quisque molestie congue venenatis. Cras maximus lorem quis mi porttitor sodales et id sapien. "
            * 246
        )
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        # assert length is byte exceed limit
        assert len(text.encode("utf-8")) > max, (
            "test input should be more than provider max limit of "
            f"{max} {unit}, got {len(text.encode('utf-8'))}"
        )
        args[input_field] = text

        with pytest.raises(ProviderException) as exc:
            compute_output("google", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputTextLengthError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputTextLengthError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

        # should not raise an error this time
        args[input_field] = text = text[:10]
        assert len(text.encode("utf-8")) < max, (
            "test input should be less than provider max limit of "
            f"{max} {unit}, got {len(text.encode('utf-8'))}"
        )
        response = compute_output("google", feature, subfeature, args)
        assert response is not None

    def test_invalid_input_file(self):
        error = microsoft_errors[ProviderInvalidInputFileError][0]
        input_field = "file"
        feature = "image"
        subfeature = "object_detection"
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        args[input_field] = resume_parser_arguments(provider_name=None)[input_field]

        with pytest.raises(ProviderException) as exc:
            compute_output("microsoft", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputFileError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputFileError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

    def test_invalid_input_file_size(self):
        error = microsoft_errors[ProviderInvalidInputFileSizeError][0]
        input_field = "file"
        feature = "ocr"
        subfeature = "ocr"
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        image_path = f"{image_data_path}/3200x2400.png"

        mime_type = mimetypes.guess_type(image_path)[0]
        file_info = FileInfo(
            os.stat(image_path).st_size,
            mime_type,
            [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        )
        file_wrapper = FileWrapper(image_path, "", file_info)
        args[input_field] = file_wrapper

        with pytest.raises(ProviderException) as exc:
            compute_output("microsoft", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputFileSizeError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputFileSizeError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

    def test_invalid_input_file_resolution(self):
        error = microsoft_errors[ProviderInvalidInputImageResolutionError][1]
        input_field = "file"
        feature = "ocr"
        subfeature = "ocr"
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        image_path = f"{image_data_path}/32x24.jpg"

        mime_type = mimetypes.guess_type(image_path)[0]
        file_info = FileInfo(
            os.stat(image_path).st_size,
            mime_type,
            [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
        )
        file_wrapper = FileWrapper(image_path, "", file_info)
        args[input_field] = file_wrapper

        with pytest.raises(ProviderException) as exc:
            compute_output("microsoft", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputImageResolutionError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputImageResolutionError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

    # def test_missing_input(self):
    #     error = ibm_errors[ProviderMissingInputError][0]
    #     input_field = "target_language"
    #     feature = "translation"
    #     subfeature = "automatic_translation"
    #     args = load_feature(
    #         FeatureDataEnum.SAMPLES_ARGS,
    #         feature=feature,
    #         subfeature=subfeature,
    #     )

    #     args[input_field] = None

    #     with pytest.raises(ProviderException) as exc:
    #         compute_output("ibm", feature, subfeature, args)

    #     assert exc.type == ProviderMissingInputError, (
    #         "exception raised wasn't the right one, expected "
    #         f"{ProviderMissingInputError.__name__} got {exc.type}"
    #     )
    #     assert (
    #         re.search(error, str(exc.value)) is not None
    #     ), f"didn't raise the right error, should be {error}, got {exc.value}"

    # def test_missing_input(self):
    #     error = ibm_errors[ProviderMissingInputError][0]
    #     input_field = "target_language"
    #     feature = "translation"
    #     subfeature = "automatic_translation"
    #     args = load_feature(
    #         FeatureDataEnum.SAMPLES_ARGS,
    #         feature=feature,
    #         subfeature=subfeature,
    #     )

    #     args[input_field] = None

    #     with pytest.raises(ProviderException) as exc:
    #         compute_output("ibm", feature, subfeature, args)

    #     assert exc.type == ProviderMissingInputError, (
    #         "exception raised wasn't the right one, expected "
    #         f"{ProviderMissingInputError.__name__} got {exc.type}"
    #     )
    #     assert (
    #         re.search(error, str(exc.value)) is not None
    #     ), f"didn't raise the right error, should be {error}, got {exc.value}"

    # def test_parsing_error(self):
    #     error = google_errors[ProviderParsingError][0]
    #     input_field = "text"
    #     feature = "translation"
    #     subfeature = "automatic_translation"
    #     args = load_feature(
    #         FeatureDataEnum.SAMPLES_ARGS,
    #         feature=feature,
    #         subfeature=subfeature,
    #     )

    #     args[input_field] = "bidou bidou ba"
    #     args["source_language"] = None

    #     with pytest.raises(ProviderException) as exc:
    #         compute_output("ibm", feature, subfeature, args)

    #     assert exc.type == ProviderParsingError, (
    #         "exception raised wasn't the right one, expected "
    #         f"{ProviderParsingError.__name__} got {exc.type}"
    #     )
    #     assert (
    #         re.search(error, str(exc.value)) is not None
    #     ), f"didn't raise the right error, should be {error}, got {exc.value}"

    def test_invalid_file_format_error(self):
        error = amazon_errors[ProviderInvalidInputFileFormatError][1]
        input_field = "file"
        feature = "image"
        subfeature = "object_detection"
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        args[input_field] = resume_parser_arguments(provider_name=None)[input_field]

        with pytest.raises(ProviderException) as exc:
            compute_output("amazon", feature, subfeature, args)

        assert exc.type == ProviderInvalidInputFileFormatError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputFileFormatError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"

    def test_invalid_file_audio_duration(self):
        error = amazon_errors[ProviderInvalidInputAudioDurationError][0]
        input_field = "file"
        feature = "audio"
        subfeature = "speech_to_text_async"
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
        )

        audio_path = f"{audio_data_path}/small.mp3"
        mime_type = mimetypes.guess_type(audio_path)[0]
        file_info = FileInfo(
            os.stat(audio_path).st_size,
            mime_type,
            [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)],
            mediainfo(audio_path).get("sample_rate", "44100"),
            mediainfo(audio_path).get("channels", "1"),
        )
        file_wrapper = FileWrapper(audio_path, "", file_info)
        args[input_field] = file_wrapper

        response = compute_output("amazon", feature, subfeature, args)
        job_id = response["provider_job_id"]

        sleep(5)

        with pytest.raises(ProviderException) as exc:
            get_async_job_result("amazon", feature, subfeature, job_id)

        assert exc.type == ProviderInvalidInputAudioDurationError, (
            "exception raised wasn't the right one, expected "
            f"{ProviderInvalidInputAudioDurationError.__name__} got {exc.type}"
        )
        assert (
            re.search(error, str(exc.value)) is not None
        ), f"didn't raise the right error, should be {error}, got {exc.value}"
