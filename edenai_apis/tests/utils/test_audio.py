from io import BytesIO
import mimetypes
from unittest.mock import Mock, patch
import pytest
import magic
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from edenai_apis.utils.audio import audio_converter, get_audio_attributes

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
            pytest.fail(f'The conversion from `wav` to `mp3` doesn\'t works')

    def test_convert_mp3_to_wav(self):
        wav_file, _, _, _ = audio_converter(self.mp3_file_input, export_format='wav')
        try:
            AudioSegment.from_file(wav_file, 'wav')
        except CouldntDecodeError:
            pytest.fail(f'The conversion from `mp3` to `wav` doesn\'t works')

    def test_convert_wav_to_wav(self):
        wav_file, _, _, _ = audio_converter(self.wav_file_input, export_format='wav')
        try:
            AudioSegment.from_file(wav_file, 'wav')
        except CouldntDecodeError:
            pytest.fail(f'The conversion from `wav` to `wav` doesn\'t works')

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
    def test_get_audio_attributes(self):
        # Arrange
        file_features = {"channels": "2", "sample_rate": "48000"}
        with patch("pydub.utils.mediainfo") as mocked_mediainfo:
            mocked_mediainfo.return_value = file_features
            with open('edenai_apis/features/audio/data/out.wav', 'rb') as audio_file:
                # Act
                channels, sample_rate = get_audio_attributes(audio_file)
                # Assert
                mocked_mediainfo.assert_called_once_with(audio_file.name)
                assert channels == 2
                assert sample_rate == 48000
