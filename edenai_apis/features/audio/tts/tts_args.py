from typing import Dict


def tts_arguments(provider_name: str) -> Dict:
    """Return test arguments for the tts subfeature.

    Args:
        provider_name: Name of the provider to get arguments for.

    Returns:
        Dictionary of test arguments.
    """
    return {
        "text": "Hello, this is a test of text to speech synthesis.",
        "voice": None,  # Uses provider default if not specified
        "audio_format": "mp3",
        "speed": 1.0,
        "provider_params": {},
    }
