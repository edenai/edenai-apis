import re
from typing import Optional


def check_resolution_format(resolution: str) -> bool:
    """Checks if the resolution is formatted correctly (IntxInt)"""
    if resolution is None:
        return None
    return bool(
        re.fullmatch(
            r"\d+x\d+",
            resolution,
        )
    )


def convert_separator(resolution: str) -> Optional[str]:
    """Converts string resolution separator to x if possible"""
    if resolution is None:
        return None

    # Use a regular expression to replace all non-digit characters with 'x'
    resolution = re.sub(r"\D", "x", resolution)
    return resolution


def provider_appropriate_resolution(resolution: str):
    # Lower case all string
    resolution = resolution.lower()

    # Convert separator to x eg : 512*512 -> 512x512
    resolution = convert_separator(resolution)

    if not check_resolution_format(resolution):
        raise SyntaxError(f"Resolution '{resolution}' badly formatted")

    return resolution
