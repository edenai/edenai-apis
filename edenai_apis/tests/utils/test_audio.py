from io import BytesIO
from pytest_mock import MockerFixture
import pytest
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from edenai_apis.utils.audio import (
    audio_converter,
    get_audio_attributes,
    audio_format,
    supported_extension,
    file_with_good_extension
)
from edenai_apis.utils.exception import ProviderException

class TestAudioConverter:
    #TODO Need to add more test, like test with other file format (m4a, flac, ogg, etc ..)
    wav_file_input = None
    mp3_file_input = None
    txt_file = None

    def setup_method(self):
        self.wav_file_input = open('edenai_apis/features/audio/data/out.wav', 'rb')
        self.mp3_file_input = open('edenai_apis/features/audio/data/conversation.mp3', 'rb')
        self.txt_file = open('edenai_apis/features/audio/data/test.txt', 'rb')

    def teardown_method(self):
        self.wav_file_input.close()
        self.mp3_file_input.close()
        self.txt_file.close()

    def test_convert_wav_to_mp3(self):
        mp3_file, _, _, _ = audio_converter(self.wav_file_input, export_format='mp3')
        try:
            AudioSegment.from_file(mp3_file, 'mp3')
        except CouldntDecodeError:
            pytest.fail('The conversion from `wav` to `mp3` doesn\'t works')

    def test_convert_mp3_to_wav(self):
        wav_file, _, _, _ = audio_converter(self.mp3_file_input, export_format='wav')
        try:
            AudioSegment.from_file(wav_file, 'wav')
        except CouldntDecodeError:
            pytest.fail('The conversion from `mp3` to `wav` doesn\'t works')

    def test_convert_wav_to_wav(self):
        wav_file, _, _, _ = audio_converter(self.wav_file_input, export_format='wav')
        try:
            AudioSegment.from_file(wav_file, 'wav')
        except CouldntDecodeError:
            pytest.fail('The conversion from `wav` to `wav` doesn\'t works')

    def test_invalid_file_format(self):
        output = audio_converter(self.txt_file)
        expected_output = None
        assert output == expected_output, \
            'Expected `None` for convertion of invalid file format, but got a `BufferReader`'

    def test_set_frame_rate_mp3_format(self):
        expected_framerate = 16000
        _, frame_rate, _, _ = audio_converter(
            audio_file=self.mp3_file_input,
            export_format='mp3',
            frame_rate=expected_framerate
        )
        assert frame_rate == expected_framerate, \
            f'Expected {expected_framerate} but got {frame_rate}'

    def test_set_channels_mp3_format(self):
        expected_channels = 1
        _, _, _, channels = audio_converter(
            audio_file=self.mp3_file_input,
            export_format='mp3',
            channels=expected_channels
        )
        assert channels == expected_channels, \
            f'Expected {expected_channels} but got {channels}'

    def test_default_frame_rate_mp3_format(self):
        mp3_file, frame_rate, _, _ = audio_converter(self.mp3_file_input, export_format='mp3')
        mp3_file_origin = AudioSegment.from_file(mp3_file, format='mp3')
        expected_framerate = mp3_file_origin.frame_rate
        assert frame_rate == expected_framerate, \
            f'Expected {expected_framerate} but got {frame_rate}'

    def test_default_channels_mp3_format(self):
        mp3_file, _, _, channels = audio_converter(self.mp3_file_input, export_format='mp3')
        mp3_file_origin = AudioSegment.from_file(mp3_file, format='mp3')
        expected_channels = mp3_file_origin.channels
        assert channels == expected_channels, \
            f'Expected {expected_channels} but got {channels}'

class TestGetAudioAttributes:
    def test_get_audio_attributes_with_good_attr(self, mocker: MockerFixture):
        def fake_mediainfo(*args, **kwargs):
            return {"channels": "2", "sample_rate": "48000"}
        # Create mock
        mocker.patch("edenai_apis.utils.audio.mediainfo", \
            side_effect=fake_mediainfo)

        with open('edenai_apis/features/audio/data/out.wav', 'rb') as audio_file:
            # Action
            channels, sample_rate = get_audio_attributes(audio_file)

            # Assert
            assert channels == 2
            assert sample_rate == 48000

    def test_get_audio_attributes_without_channels(self, mocker: MockerFixture):
        def fake_mediainfo(*args, **kwargs):
            return {"sample_rate": "48000"}
        # Create mock
        mocker.patch("edenai_apis.utils.audio.mediainfo", \
            side_effect=fake_mediainfo)

        with open('edenai_apis/features/audio/data/out.wav', 'rb') as audio_file:
            # Action
            channels, sample_rate = get_audio_attributes(audio_file)

            # Assert
            assert channels == 1
            assert sample_rate == 48000

    def test_get_audio_attributes_without_sample_rate(self, mocker: MockerFixture):
        def fake_mediainfo(*args, **kwargs):
            return {"channels": "2"}
        # Create mock
        mocker.patch("edenai_apis.utils.audio.mediainfo", \
            side_effect=fake_mediainfo)

        with open('edenai_apis/features/audio/data/out.wav', 'rb') as audio_file:
            # Action
            channels, sample_rate = get_audio_attributes(audio_file)

            # Assert
            assert channels == 2
            assert sample_rate == 44100

class TestAudioFormat:
    def test_valid_file_type(self):
        audio_file = BytesIO(b"test file")
        audio_file.name = "audio.mp3"
        assert audio_format(audio_file)[0] == "mp3"

    def test_unknown_file_type(self):
        audio_file = BytesIO(b"test file")
        audio_file.name = "audio.xyz"
        assert audio_format(audio_file) == ["xyz"]

class TestSupportedExtension:
    def test_valid_extension(self):
        file = BytesIO(b"test file")
        file.name = "audio.mp3"
        accepted_extensions = ["mp3", "wav", "m4a"]
        assert supported_extension(file, accepted_extensions) == (True, "mp3")

    def test_valid_extension_multiple(self):
        file = BytesIO(b"test file")
        file.name = "audio.mp3"
        accepted_extensions = ["mp3", "mp4", "m4a"]
        assert supported_extension(file, accepted_extensions) == (True, "mp3")

    def test_invalid_extension(self):
        file = BytesIO(b"test file")
        file.name = "audio.xyz"
        accepted_extensions = ["mp3", "wav", "m4a"]
        assert supported_extension(file, accepted_extensions) == (False, "nop")


class TestFileWithGoodExtension:
    def test_raises_exception_on_unsupported_extension(self):
        file = BytesIO(b"fake audio file")
        file.name = 'fake.txt'
        accepted_extensions = ["wav", "mp3"]
        with pytest.raises(ProviderException) as exc:
            file_with_good_extension(file, accepted_extensions)
        assert str(exc.value) == \
            f"File extension not supported. Use one of the following extensions: {accepted_extensions}"

    def test_raises_exception_on_mismatch_channels(self):
        with open('edenai_apis/features/audio/data/out.wav', 'rb') as file:
            accepted_extensions = ["wav", "mp3"]
            channels = 2
            with pytest.raises(ProviderException) as exc:
                file_with_good_extension(file, accepted_extensions, channels)
            assert str(exc.value) == "File audio must be Stereo"

    def test_raises_exception_on_mismatch_channels_mp3(self):
        with open('edenai_apis/features/audio/data/conversation.mp3', 'rb') as file:
            accepted_extensions = ["wav", "mp3"]
            channels = 1
            with pytest.raises(ProviderException) as exc:
                file_with_good_extension(file, accepted_extensions, channels)
            assert str(exc.value) == "File audio must be Mono"
