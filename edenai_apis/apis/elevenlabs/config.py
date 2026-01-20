from typing import Tuple

# Default ElevenLabs output format mappings
# Format: codec_samplerate_bitrate (e.g., mp3_44100_128)
# See: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
ELEVENLABS_FORMAT_MAP = {
    "mp3": "mp3_44100_128",
    "wav": "pcm_44100",  # ElevenLabs uses pcm for wav-like output
    "pcm": "pcm_44100",
    "opus": "opus_48000_128",
    "alaw": "alaw_8000",
    "ulaw": "ulaw_8000",
}

DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


def get_audio_format_and_extension(audio_format: str) -> Tuple[str, str]:
    """Convert simple audio format to ElevenLabs format and file extension.

    Args:
        audio_format: Simple format like "mp3", "wav", or full ElevenLabs
                      format like "mp3_44100_128"

    Returns:
        Tuple of (elevenlabs_format, file_extension)
        e.g., ("mp3_44100_128", "mp3") or ("pcm_44100", "wav")
    """
    if not audio_format:
        return DEFAULT_OUTPUT_FORMAT, "mp3"

    # Check if it's already a full ElevenLabs format (contains underscores with numbers)
    if "_" in audio_format and any(c.isdigit() for c in audio_format):
        # Extract codec (first part before underscore) for extension
        codec = audio_format.split("_")[0]
        # Map pcm to wav for file extension
        extension = "wav" if codec == "pcm" else codec
        return audio_format, extension

    # Map simple format to ElevenLabs format
    elevenlabs_format = ELEVENLABS_FORMAT_MAP.get(audio_format.lower(), DEFAULT_OUTPUT_FORMAT)

    # Determine file extension
    if audio_format.lower() in ("pcm", "wav"):
        extension = "wav"
    else:
        extension = audio_format.lower()

    return elevenlabs_format, extension
