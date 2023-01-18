import io
import pytest
import wave
from typing import Union, Optional, Callable, Tuple
from pydub import AudioSegment

from edenai_apis.utils.audio import audio_converter

AUDIO_FILE_FORMAT = ["wav", "flac", "mp3", "flv", "ogg", "wma", "mp4", "aac", "m4a"]

class TestAudioConverter:
    file_test = 'edenai_apis/feature/audio/data/out.wav'
    def test_invalid_file_format(self):
        invalid_file = io.BytesIO(b"invalid file content")
        invalid_file.name = "invalid_file.invalid"
        assert audio_converter(invalid_file) == None

    def test_valid_file_format(self):
        with wave.open(self.file_test, "rb") as valid_file:
            valid_file.setsampwidth(2)
            valid_file.setnchannels(1)
            valid_file.setframerate(44100)
        valid_file = io.BytesIO(valid_file.readframes(valid_file.getnframes()))
        valid_file.name = self.file_test
        new_audio, frame_rate, frame_width, channels = audio_converter(valid_file)
        assert new_audio.format == "wav"
        assert frame_rate == 44100
        assert frame_width == 2
        assert channels == 1

    def test_different_export_format(self):
        with wave.open(self.file_test, "rb") as valid_file:
            valid_file.setsampwidth(2)
            valid_file.setnchannels(1)
            valid_file.setframerate(44100)
        valid_file = io.BytesIO(valid_file.readframes(valid_file.getnframes()))
        valid_file.name = self.file_test
        new_audio, _, _, _ = audio_converter(valid_file, export_format="mp3")
        assert new_audio.format == "mp3"
        new_audio, _, _, _ = audio_converter(valid_file, export_format="ogg")
        assert new_audio.format == "ogg"

    def test_different_frame_rate(self):
        with wave.open(self.file_test, "rb") as valid_file:
            valid_file.setsampwidth(2)
            valid_file.setnchannels(1)
            valid_file.setframerate(44100)
        valid_file = io.BytesIO(valid_file.readframes(valid_file.getnframes()))
        valid_file.name = self.file_test
        _, frame_rate, _, _ = audio_converter(valid_file, frame_rate=48000)
        assert frame_rate == 48000

    def test_different_number_of_channels(self):
        with wave.open(self.file_test, "rb") as valid_file:
            valid_file.setsampwidth(2)
            valid_file.setnchannels(1)
            valid_file.setframerate(44100)
        valid_file = io.BytesIO(valid_file.readframes(valid_file.getnframes()))
        valid_file.name = self.file_test
        _, _, _, channels = audio_converter(valid_file, channels=2)
        assert channels == 2
