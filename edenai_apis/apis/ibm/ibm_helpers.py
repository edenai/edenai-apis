from edenai_apis.utils.audio import validate_audio_attribute_against_ssml_tags_use
from edenai_apis.utils.exception import AsyncJobException, AsyncJobExceptionReason, ProviderException
from edenai_apis.utils.ssml import (
    convert_audio_attr_in_prosody_tag,
    get_index_after_first_speak_tag,
    get_index_before_last_speak_tag,
    is_ssml,
)
from watson_developer_cloud.watson_service import WatsonApiException

def handle_ibm_call(function_call, **kwargs):
    provider_job_id_error = "job not found"
    try:
        response = function_call(**kwargs)
    except WatsonApiException as exc:
        message = exc.message
        code = exc.code
        if provider_job_id_error in str(exc):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID,
                code = code
            )
        raise ProviderException(message, code = code)
    except Exception as exc:
        if provider_job_id_error in str(exc):
            raise AsyncJobException(
                reason=AsyncJobExceptionReason.DEPRECATED_JOB_ID
            )
        raise ProviderException(str(exc))
    return response

def generate_right_ssml_text(text, speaking_rate, speaking_pitch):
    attribs = {"rate": speaking_rate, "pitch": speaking_pitch}
    cleaned_attribs_string = ""
    for k, v in attribs.items():
        if not v:
            continue
        cleaned_attribs_string = f"{cleaned_attribs_string} {k}='{v}%'"
    if not cleaned_attribs_string.strip():
        return text
    return convert_audio_attr_in_prosody_tag(cleaned_attribs_string, text)


# list of audio format with there extension, Wether or not a sampling rate is required, and also if you can specify
# a sampling rate with any value or from a list of samplings
# Exmp: ("mp3", "mp3", False, []) => means the audio format mp3 with an extension mp3 with no required sampling value and
# can optionnaly have any sampling rate between 8000Hz and 192000Hz
audio_format_list_extensions = [
    ("alaw", "alaw", True, []),
    ("basic", "basic", False, None),
    ("flac", "flac", False, []),
    ("l16", "pcm", True, []),
    ("mp3", "mp3", False, []),
    ("mulaw", "mulaw", True, []),
    ("ogg", "ogg", False, []),
    ("ogg-opus", "ogg", False, [48000, 24000, 16000, 12000, 8000]),
    ("ogg-vorbis", "ogg", False, []),
    ("wav", "wav", False, []),
    ("webm", "webm", False, None),
    ("webm-opus", "webm", False, None),
    ("webm-vorbis", "webm", False, []),
]


def get_right_audio_support_and_sampling_rate(audio_format: str, sampling_rate: int):
    if sampling_rate and (sampling_rate < 8000 or sampling_rate > 192000):
        raise ProviderException(
            "Sampling rate must lie in the range of 8 kHz to 192 kHz"
        )
    if not audio_format:
        audio_format = "mp3"
    right_audio_format = next(
        filter(lambda x: x[0] == audio_format, audio_format_list_extensions), None
    )
    file_extension = right_audio_format[1]
    audio_format = audio_format.replace("-", ";codecs=")
    audio_format = f"audio/{audio_format}"
    if not sampling_rate:  # no sampling provided
        if right_audio_format[2]:
            raise ProviderException(
                f"You must specify a sampling rate for the '{audio_format}' audio format"
            )
        return file_extension, audio_format
    if right_audio_format[3] is None:  # you can not specify a sampling rate value
        return file_extension, audio_format
    if isinstance(right_audio_format[3], list) and len(right_audio_format[3]) > 0:
        # get the nearest
        nearest_rate = min(right_audio_format[3], key=lambda x: abs(x - sampling_rate))
        return file_extension, f"{audio_format};rate={nearest_rate}"
    return file_extension, f"{audio_format};rate={sampling_rate}"
