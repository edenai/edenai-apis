"""Utility functions for Text-to-Speech (TTS) operations."""


def normalize_speed_for_ssml(speed: float) -> str:
    """Convert speed multiplier (0.5-2.0) to SSML percentage format.

    Args:
        speed: Speed multiplier where 1.0 is normal speed.
               Range: 0.5 (half speed) to 2.0 (double speed)

    Returns:
        SSML percentage string (e.g., "100%", "150%", "50%")
    """
    if speed is None:
        return "100%"
    # Clamp to valid range
    speed = max(0.5, min(2.0, speed))
    percentage = int(speed * 100)
    return f"{percentage}%"


def normalize_speed_for_openai(speed: float) -> float:
    """Clamp speed to OpenAI's valid range (0.25-4.0).

    Args:
        speed: Speed multiplier where 1.0 is normal speed.

    Returns:
        Speed value clamped to OpenAI's 0.25-4.0 range.
    """
    if speed is None:
        return 1.0
    return max(0.25, min(4.0, speed))


def normalize_speed_for_lovoai(speed: float) -> float:
    """Clamp speed to LovoAI's valid range (0.5-1.5).

    Args:
        speed: Speed multiplier where 1.0 is normal speed.

    Returns:
        Speed value clamped to LovoAI's 0.5-1.5 range.
    """
    if speed is None:
        return 1.0
    return max(0.5, min(1.5, speed))
