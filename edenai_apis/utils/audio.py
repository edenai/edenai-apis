import mimetypes
from io import BufferedReader
from typing import Union, List, Tuple
from pydub import AudioSegment
from pydub.utils import mediainfo
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException

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
    return int(file_features.get("channels", "1")), int(file_features.get("sample_rate", "44100"))


def audio_format(audio_file: BufferedReader):
    mime_type, _ = mimetypes.guess_type(audio_file.name)
    extensions = [extension[1:] for extension in mimetypes.guess_all_extensions(mime_type)]
    if not extensions:
        file_name = str(audio_file.name.split("/")[-1])
        index_extension = str(file_name.split(".")[-1])
        return [index_extension]
    return extensions


def supported_extension(file, accepted_extensions: List):
    extensions = audio_format(file)
    if len(extensions) == 1:
        if extensions[0] in accepted_extensions:
            return True, *extensions
        return False, "nop"
    if len(extensions) > 1:
        file_name = file.name.split("/")[-1]
        for extension in accepted_extensions:
            if file_name.endswith(extension):
                return True, extension
        return False, "nop"
    return False, "nop"

def __channel_number_to_str(channel_number):
    if channel_number == 1:
        return "Mono"
    return "Stereo"

def file_with_good_extension(
    file: BufferedReader,
    accepted_extensions: List,
    channels: int= None) -> Tuple[BufferedReader, str, int, int]:
    """ Checks whether or not the extension of the audio file is within the list of accepted extensions,
    otherwise, it raise a Provider Exception for the format used

    Args:
        file (BufferedReader): the audio file
        accepted_extensions (List): a list of the accepted extentions
        channels (int, optional): To change the number of channels of the audio file(mono or stereo). Defaults to None.

    Returns:
        Tuple[BufferedReader, str, int, int]: returns the file, the export extention, the number of channels
        and the frame rate
    """
    accepted_format, export_format = supported_extension(file, accepted_extensions)
    file.seek(0)

    if not accepted_format:
        raise ProviderException(f"File extension not supported. Use one of the following extensions: {accepted_extensions}")

    audio_channels, frame_rate = get_audio_attributes(file)

    if channels and channels != audio_channels:
        raise ProviderException(f"File audio must be {__channel_number_to_str(channels)}")

    file.seek(0)
    return export_format, channels, frame_rate


#decorator
def audio_features_and_support(func):
    """decorator to pass audio features or attributes to audio__speech_to_text_async__launch_job"""
    def func_wrapper(self, file: BufferedReader, language: str,
        speakers: int, profanity_filter: bool, vocabulary: list,
        convert_wav: bool = False):
        file_name = file.name

        if convert_wav:
            try:
                wav_file, frame_rate, _, channels = audio_converter(file)
                audio_attributes = ("wav", channels, frame_rate)
                file_name = f"{'.'.join(file_name.split('.')[:-1])}.wav"
                return func(self, wav_file, file_name, language, speakers, profanity_filter, vocabulary, audio_attributes)
            except Exception as excp:
                raise ProviderException("Couldn't convert audio file to wav..")

        provider_name = getattr(self, "provider_name")
        info_file = load_provider(ProviderDataEnum.INFO_FILE, provider_name)
        audio_feature = info_file.get("audio") or {}
        speech_to_text_subfeature = audio_feature.get("speech_to_text_async") or {}
        constraints = speech_to_text_subfeature.get("constraints", {})
        accepted_extensions : list = constraints.get("file_extensions", []) or []
        audio_attributes = file_with_good_extension(file, accepted_extensions)
        return func(self, file, file_name, language, speakers, profanity_filter, vocabulary, audio_attributes)
    return func_wrapper
