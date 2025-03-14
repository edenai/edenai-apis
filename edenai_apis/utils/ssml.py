import re
from typing import Optional


def is_ssml(ssml_text: str) -> bool:
    """Check if the text is ssml_text.
    Check if <speak> tag is present. with regex
    Args:
        ssml_text (str): text to check

    Returns:
        bool: True if ssml_text, False otherwise
    """
    regex: str = (
        r"^<\s*speak\b[^>]*>(?:(?!(<\/\s*speak\s*>|<speak\s*>))[\s\S])*<\/\s*speak\s*>$"
    )
    match: Optional[re.Match] = re.match(regex, ssml_text, re.MULTILINE)
    return match is not None


def get_index_after_first_speak_tag(ssml_text) -> int:
    """Get the index after the first <speak> tag.

    Args:
        ssml_text (str): text to check

    Returns:
        int: index after the first <speak> tag. -1 if not found
    """
    if not is_ssml(ssml_text):
        return -1
    regex = r"<\s*speak\b[^>]*>"
    match = re.search(regex, ssml_text)

    if match:
        return match.end()

    return -1


def get_index_before_last_speak_tag(ssml_text) -> int:
    """Get the index before the last </speak> tag.
    Args:
        ssml_text (str): text to check
    Returns:
        int: index before the last </speak> tag. -1 if not found
    """
    if not is_ssml(ssml_text):
        return -1

    regex = r"<\s*\/\s*speak\s*>"
    match = re.search(regex, ssml_text)
    if match:
        return match.start()
    return -1


def convert_audio_attr_in_prosody_tag(
    cleaned_attribs: str,
    text: str,
    voice_tag: str = "",
    speak_attr: Optional[str] = None,
) -> str:
    """Convert pitch, volume and rate in attribute for prosody tag

    Args:
        cleaned_attribs (str): The format string of audio attribute
        text (str): The input text for tts
        voice_id (Optional[str]): The voice id for tts. Defaults to "".
        speak_attr (Optional[str]): The speak attribute for tts. Defaults to None. Ignore if text already have <speak> tag.
    """
    idx_after_first_tag = get_index_after_first_speak_tag(text)
    idx_before_last_tag = get_index_before_last_speak_tag(text)

    if idx_after_first_tag == -1 or idx_before_last_tag == -1:
        escaped_text = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            f"<speak{f' {speak_attr}' if speak_attr else ''}>"
            + f"{voice_tag}"
            + (f"<prosody {cleaned_attribs}>" if cleaned_attribs else "")
            + escaped_text
            + (f"</prosody>" if cleaned_attribs else "")
            + f"{f'</voice>' if voice_tag else ''}</speak>"
        )
    return (
        text[0:idx_after_first_tag]
        + f"{voice_tag}"
        + (f"<prosody {cleaned_attribs}>" if cleaned_attribs else "")
        + text[idx_after_first_tag:idx_before_last_tag]
        + (f"</prosody>" if cleaned_attribs else "")
        + f"{f'</voice>' if voice_tag else ''}"
        + text[idx_before_last_tag:]
    )
