from io import BufferedReader
from typing import Union, List, Tuple
import magic
import mimetypes
from enum import Enum

from pydub import AudioSegment
from edenai_apis.utils.exception import ProviderException



def channel_number_to_str(channel_number):
    if channel_number == 1:
        return "Mono"
    return "Stereo"


def wav_converter(
    audio_file: BufferedReader,
    export_format: str = "wav",
    frame_rate: Union[int, None] = None,
    channels: Union[int, None] = None,
):
    """
    :param audio_path:      Path of the file that need to be converted
    :return:                New path
    """
    file_extension = audio_file.name.split(".")[-1]

    if file_extension in [
        "wav",
        "flac",
        "mp3",
        "flv",
        "ogg",
        "wma",
        "mp4",
        "aac",
        "m4a",
    ]:
        # output_path = '.'.join(audio_path.split('.')[:-1]) + '.wav'
        audio_out: AudioSegment = AudioSegment.from_file(
            audio_file, format=file_extension
        )
        # file.export(output_path, format='wav')
        if frame_rate:
            # print(audio_out.frame_rate)
            audio_out = audio_out.set_frame_rate(frame_rate)
            # print(audio_out.frame_rate)
        if channels:
            print(channels)
            # print(audio_out.frame_rate)
            if audio_out.channels != channels:
                audio_out = audio_out.set_channels(channels)
        # audio_out = audio_out.set_channels(1)
        return (
            audio_out.export(format=export_format),
            audio_out.frame_rate,
            audio_out.frame_width,
            audio_out.channels,
        )

    else:
        return None

def get_audio_attributes(audio_file: BufferedReader, export_format:str):
    audio_file.seek(0)
    audio_out: AudioSegment = AudioSegment.from_file(audio_file)
    return audio_out.channels, audio_out.frame_rate


def audio_format(audio_file: BufferedReader):
    mgi = magic.Magic(mime=True)
    mtpe = mgi.from_buffer(audio_file.read())
    extensions = [extension[1:] for extension in mimetypes.guess_all_extensions(mtpe)]
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


def file_with_good_extension(file: BufferedReader, accepted_extensions: List, 
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
    accepte_format, export_format = supported_extension(file, accepted_extensions)
    file.seek(0)
    if not accepte_format:
        raise ProviderException(f"File extension not supported. Use one of the following extension: {accepted_extensions}")
    if channels:
        audio_channels, _ = get_audio_attributes(file, export_format) 
        if channels != audio_channels:
            raise ProviderException(f"File audio must be {channel_number_to_str(channels)}")
 
    channels, frame_rate = get_audio_attributes(file, export_format)
    file.seek(0)
    return file, export_format, channels, frame_rate
