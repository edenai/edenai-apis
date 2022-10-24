from io import BufferedReader
from typing import Union

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
            # print(audio_out.frame_rate)
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
