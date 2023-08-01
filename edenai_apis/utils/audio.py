import mimetypes
import random
from io import BufferedReader
from typing import Union, List, Tuple, Dict
from pydub import AudioSegment
from pydub.utils import mediainfo
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.files import FileWrapper
from edenai_apis.utils.languages import provide_appropriate_language

VOICE_EXCEPTION_MESSAGE = "Wrong voice id"
SSML_TAG_EXCEPTION_MESSAGE = (
    "Remove audio attributes 'rate, pitch or volume' to be able to use ssml tags, or "
    "add them manually using tags."
)

AUDIO_FILE_FORMAT = [
    "wav",
    "flac",
    "mp3",
    "flv",
    "ogg",
    "wma",
    "mp4",
    "aac",
    "m4a",
]


def audio_converter(
    audio_file: BufferedReader,
    export_format: str = "wav",
    frame_rate: Union[int, None] = None,
    channels: Union[int, None] = None,
):
    """Convert an audio file in a given format.
    Format for destination audio file. ('mp3', 'wav', 'raw', 'ogg' or other ffmpeg/avconv supported files)

    Args:
        audio_file (BufferedReader): The audio file to be converted, in the form of a BufferedReader object.
        export_format (str, optional): The format of the exported audio file. Defaults to "wav".
        frame_rate (Union[int, None], optional): The frame rate of the output audio file. Defaults to None.
        channels (Union[int, None], optional): Number of channels of the output audio file. Defaults to None.

    Returns:
        Tuple with new audio format, frame_rate, frame_width and the number of channels in audio
    """
    file_extension = audio_file.name.split(".")[-1]

    # if file_extension not in AUDIO_FILE_FORMAT:
    #     return None

    audio_out: AudioSegment = AudioSegment.from_file(audio_file, format=file_extension)

    if frame_rate:
        audio_out = audio_out.set_frame_rate(frame_rate)

    if channels and audio_out.channels != channels:
        audio_out = audio_out.set_channels(channels)

    return (
        audio_out.export(format=export_format),
        audio_out.frame_rate,
        audio_out.frame_width,
        audio_out.channels,
    )


def get_audio_attributes(audio_file: BufferedReader):
    file_features = mediainfo(audio_file.name)
    return int(file_features.get("channels", "1")), int(
        file_features.get("sample_rate", "44100")
    )


def audio_format(audio_file_path: str, extensions: List[str]):
    if not extensions:
        file_name = str(audio_file_path.split("/")[-1])
        index_extension = str(file_name.split(".")[-1])
        return [index_extension]
    return extensions


def supported_extension(file: FileWrapper, accepted_extensions: List[str]):
    extensions = audio_format(file.file_path, file.file_info.file_extension)
    if len(extensions) == 1:
        if extensions[0] in accepted_extensions:
            return True, *extensions
        return False, "nop"
    if len(extensions) > 1:
        file_name = file.file_path.split("/")[-1]
        for extension in accepted_extensions:
            if file_name.endswith(extension):
                return True, extension
        return False, "nop"
    return False, "nop"


def __channel_number_to_str(channel_number):
    if channel_number == 1:
        return "Mono"
    return "Stereo"


def get_file_extension(
    file: FileWrapper, accepted_extensions: List[str], channels: int = None
) -> str:
    """Checks whether or not the extension of the audio file is within the list of accepted extensions,
    otherwise, it raise a Provider Exception for the format used

    Args:
        file (FileWrapper): the audio file
        accepted_extensions (List): a list of the accepted extentions
        channels (int, optional): To change the number of channels of the audio file(mono or stereo). Defaults to None.

    Returns:
        str: returns the export extention
    """
    accepted_format, export_format = supported_extension(file, accepted_extensions)

    if not accepted_format:
        raise ProviderException(
            f"File extension not supported. Use one of the following extensions: {', '.join(accepted_extensions)}"
        )

    if channels and channels != int(file.file_info.file_channels):
        raise ProviderException(
            f"File audio must be {__channel_number_to_str(channels)}"
        )

    return export_format


# ******Text_to_Speech******#


def __confirm_appropriate_language(language: str, provider: str, subfeature: str):
    if not language:
        return None
    try:
        formated_language = provide_appropriate_language(
            language,
            provider_name=provider,
            feature="audio",
            subfeature=subfeature,
        )
    except SyntaxError as exc:
        formated_language = None
    if not formated_language:
        return []
    return formated_language


def __get_voices_from_constrains(constraints: Dict, language: str, gender: str):
    if isinstance(language, list):
        return []
    voices = {
        "MALE": constraints["voice_ids"]["MALE"],
        "FEMALE": constraints["voice_ids"]["FEMALE"],
    }
    if language:
        voices = {
            "MALE": list(
                filter(lambda voice: voice.startswith(language), voices["MALE"])
            ),
            "FEMALE": list(
                filter(lambda voice: voice.startswith(language), voices["FEMALE"])
            ),
        }
    if gender:
        voices = voices["MALE"] if gender.upper() == "MALE" else voices["FEMALE"]
    return voices


def __get_provider_tts_constraints(provider, subfeature):
    try:
        provider_info = load_provider(
            ProviderDataEnum.PROVIDER_INFO, provider, "audio", subfeature
        )
        if constrains := provider_info.get("constraints"):
            return constrains
    except:
        pass
    return {}


def __has_voice_in_contrains(contraints: Dict, voice: str):
    all_voices = contraints["voice_ids"]["MALE"] + contraints["voice_ids"]["FEMALE"]
    return voice in all_voices


def get_voices(language: str, subfeature: str, gender: str, providers: List[str]) -> Dict[str, List]:
    """Returns the list of voices for each provider withing the providers parameter according the the language and gender

    Args:
        language (str): The input language
        gender (str): Either MALE or FEMALE
        providers (List[str]): List of the providers

    Returns:
        Dict[str, List]: List pf voices for each provider
    """
    voices = {}
    for provider in providers:
        constrains = __get_provider_tts_constraints(provider, subfeature)
        if constrains:
            formtatted_language = __confirm_appropriate_language(language, provider, subfeature)
            voices.update(
                {
                    provider: __get_voices_from_constrains(
                        constrains, formtatted_language, gender
                    )
                }
            )
    return voices


def retreive_voice_id(
    provider_name, subfeature: str, language: str, option: str, settings: Dict = {}
) -> str:
    """Retreives a voice id for text_to_speech methods depending on the settings parameters if a voice_id is \
        specified, otherwise depening on the language and the gender

    Args:
        language (str): The input language
        gender (str): Either MALE or FEMALE
        settings (Dict, optional): Dictionnary contraining the selected voice id for each provider. Defaults to {}.

    Raises:
        ProviderException: if wrong voice id provided in the settings dictionnary
        ProviderException: if a voice is only available in the opposite gender than specified by the user

    Returns:
        str: the voice id selected
    """
    # provider_name = getattr(object_instance, "provider_name")
    constrains = __get_provider_tts_constraints(provider_name, subfeature)
    language = __confirm_appropriate_language(language, provider_name, subfeature)
    if isinstance(language, list):
        language = None
    if settings and provider_name in settings:
        selected_voice = settings[provider_name]
        if constrains and __has_voice_in_contrains(constrains, selected_voice):
            return selected_voice
        raise ProviderException(VOICE_EXCEPTION_MESSAGE)
    if not language:
        raise ProviderException(f"Language '{language}' not supported")
    suited_voices = __get_voices_from_constrains(constrains, language, option)
    if not suited_voices:
        option_supported = "MALE" if option.upper() == "FEMALE" else "FEMALE"
        raise ProviderException(
            f"Only {option_supported} voice is available for the {language} language code"
        )
    suited_voices.sort()
    return suited_voices[0]


def validate_audio_attribute_against_ssml_tags_use(text, rate, pitch, volume):
    if "<speak>" in text:
        if any((rate, pitch, volume)):
            raise ProviderException(SSML_TAG_EXCEPTION_MESSAGE)
        return True
