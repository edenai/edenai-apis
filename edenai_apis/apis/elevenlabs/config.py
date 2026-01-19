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


voice_ids = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Drew": "29vD33N1CtxCmqQRPOHJ",
    "Clyde": "2EiwWnXFnvU5JabPnv8n",
    "Paul": "5Q0t7uMcjvnagumLfvZi",
    "Dave": "CYw3kZ02Hs0563khs1Fj",
    "Fin": "D38z5RcWu1voky8WS1ja",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Antoni": "ErXwobaYiN019PkySvjV",
    "Thomas": "GBv7mTt0atIp3Br8iCZE",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Emily": "LcfcDJNUP1GQjkzn1xUU",
    "Elli": "MF3mGyEYCl7XYWbV9V6O",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Patrick": "ODq5zmih8GrVes37Dizd",
    "Harry": "SOYHLrjzK2X1ezoPC6cr",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "Dorothy": "ThT5KcBeYPX3keUQqHPh",
    "Josh": "TxGEqnHWrfWFTfGW9XjX",
    "Arnold": "VR6AewLTigWG4xSOukaG",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "Matthew": "Yko7PKHZNXotIFUBG7I9",
    "James": "ZQe5CZNOzWyzPSCn5a3c",
    "Joseph": "Zlb1dXrM653N07WRdFW3",
    "Jeremy": "bVMeCyTHy58xNoL34h3p",
    "Michael": "flq6f7yk4E4fJM5XTYuZ",
    "Ethan": "g5CIjZEefAph4nQFvHAz",
    "Gigi": "jBpfuIE2acCO8z3wKNLl",
    "Freya": "jsCqWAovK2LkecY7zXl4",
    "Santa Claus": "knrPHWnBmmDHMoiMeP3l",
    "Grace": "oWAxZDx7w5VEj9dCyTzz",
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Lily": "pFZP5JQG7iQjIQuC4Bku",
    "Serena": "pMsXgVXv3BLzUgSXRplE",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Nicole": "piTKgcLEGmPE4e6mEKli",
    "Bill": "pqHfZKP75CvOlQylNhV4",
    "Jessie": "t0jbNlBVZ17f02VDIeMI",
    "Ryan": "wViXBPUzp2ZZixB1xQuM",
    "Sam": "yoZ06aMxZJJ28mfd3POQ",
    "Glinda": "z9fAnlkpzviPz146aGWa",
    "Giovanni": "zcAOhNBS3c14rBihAFp1",
    "Mimi": "zrHiDhphv9ZnVXBqCLjz",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
}
