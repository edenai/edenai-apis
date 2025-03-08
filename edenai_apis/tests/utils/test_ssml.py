import pytest

from edenai_apis.utils.ssml import (
    convert_audio_attr_in_prosody_tag,
    get_index_after_first_speak_tag,
    get_index_before_last_speak_tag,
    is_ssml,
)


class TestIsSSML:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ssml_text",
        [
            "<speak>hello</speak>",
            "<speak version='1.0'>hello</speak>",
            '<speak> Hello <break time="3s"/> world </speak>',
            "<speak><voice-id 453>Hello world</voice-id></speak>",
            '<speak><voice-id 453>Hello <break time="3s"/> world</voice-id></speak>',
            '<speak><voice-id 453>Hello <break time="3s"/> <break time="3s"/> <break time="3s"/> world</voice-id></speak>',
            '<speak><voice-id 453><prosody pitch=0.35>Hello <break time="3s"/> world</prosody></voice-id></speak>',
        ],
    )
    def test__is_valid_ssml(self, ssml_text: str):
        assert is_ssml(ssml_text) == True, f"ssml_text `{ssml_text}` is valid ssml_text"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ssml_text",
        [
            "hello",
            "<speak>hello",
            "hello</speak>",
            "<speak>hello</speak><speak>hello</speak>",
            "<speak>hello</speak><speak>hello</speak><speak>hello</speak>",
            "<speak>hello</speak>hello",
            "hello<speak>hello</speak>",
            "<speak version='1.0'>hello</speak dsdcs>",
        ],
    )
    def test__is_invalid_ssml(self, ssml_text: str):
        assert (
            is_ssml(ssml_text) == False
        ), f"ssml_text `{ssml_text}` is not valid ssml_text"


class TestGetIndexAfterFirstSpeakTag:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ssml_text, expected",
        [
            ("<speak>hello</speak>", 7),
            ("<speak version='1.0'>hello</speak>", 21),
        ],
    )
    def test__get_index_after_first_speak_tag(self, ssml_text: str, expected: int):
        assert (
            get_index_after_first_speak_tag(ssml_text) == expected
        ), f"ssml_text `{ssml_text}` should return index {expected} after first <speak> tag"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ssml_text",
        [
            "hello",
            "<speak>hello",
            "hello</speak>",
            "<speak>hello</speak>hello",
            "hello<speak>hello</speak>",
            "<speak>hello</speak><speak>hello</speak>",
            "<speak>hello</speak><speak>hello</speak><speak>hello</speak>",
        ],
    )
    def test__get_index_after_first_speak_tag__invalid_ssml(self, ssml_text: str):
        assert (
            get_index_after_first_speak_tag(ssml_text) == -1
        ), f"ssml_text `{ssml_text}` should return index -1 after first <speak> tag"


class TestGetIndexBeforeLastSpeakTag:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ssml_text, expected",
        [
            ("<speak>hello</speak>", 12),
            ("<speak version='1.0'>hello</speak>", 26),
        ],
    )
    def test__get_index_before_last_speak_tag(self, ssml_text: str, expected: int):
        assert (
            get_index_before_last_speak_tag(ssml_text) == expected
        ), f"ssml_text `{ssml_text}` should return index {expected} before last </speak> tag"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ssml_text",
        [
            "hello",
            "<speak>hello",
            "hello</speak>",
            "<speak>hello</speak>hello",
            "hello<speak>hello</speak>",
            "<speak>hello</speak><speak>hello</speak>",
            "<speak>hello</speak><speak>hello</speak><speak>hello</speak>",
        ],
    )
    def test__get_index_before_last_speak_tag__invalid_ssml(self, ssml_text: str):
        assert (
            get_index_before_last_speak_tag(ssml_text) == -1
        ), f"ssml_text `{ssml_text}` should return index -1 before last </speak> tag"


class TestConvertAudioAttriInProsodyTag:
    @pytest.mark.unit
    def test__convert_audio_attri_in_prosody_tag(self):
        cleaned_attribs = "pitch=0.35"
        text = "Hello world"
        expected = "<speak><prosody pitch=0.35>Hello world</prosody></speak>"

        assert (
            convert_audio_attr_in_prosody_tag(cleaned_attribs, text) == expected
        ), f"should return {expected} after converting audio attributes in prosody tag"
