from io import BufferedReader
from typing import Union, List
import magic
import mimetypes

from pydub import AudioSegment


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


def file_with_good_extension(file: BufferedReader, accepted_extensions: List, channels: int= None):
    accepte_format, export_format = supported_extension(file, accepted_extensions)
    file.seek(0)
    file_config = {
            "audio_file": file,
            "export_format" : export_format,
        }
    if not accepte_format:
        export_format = accepted_extensions[0] #take first element as the one to choose if the extension is not accepted
        file_config["export_format"] = export_format
        file, frame_rate, _, channels = wav_converter(**file_config, channels=channels)
    else:
        if channels:
            file_config.update({
                "export_format": export_format,
                "channels": channels
            })
            file, frame_rate, _, channels = wav_converter(**file_config)
        else: 
            channels, frame_rate = get_audio_attributes(file, export_format)
    file.seek(0)
    return file, export_format, channels, frame_rate
