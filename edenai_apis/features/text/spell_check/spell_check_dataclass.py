from typing import Optional, Sequence
from pydantic import BaseModel, Field, validator


class SuggestionItem(BaseModel):
    """
    Represents a suggestion for a misspelled word.

    Args:
        suggestion (str): The suggested text.
        score (float, optional): The score of the suggested text (between 0 and 1).

    Raises:
        ValueError: If the score is not between 0 and 1.

    Returns:
        SuggestionItem: An instance of the SuggestionItem class.
    """
    suggestion: str
    score: Optional[float] = Field(default=None)

    @validator('score')
    def score_must_be_between_0_and_1(cls, v: float) -> float:
        """
        Validates if the score is between 0 and 1, and rounds it to 2 decimal places.

        Args:
            v (float): The score to validate.

        Raises:
            ValueError: If the score is not between 0 and 1.

        Returns:
            float: The rounded score.
        """
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Score must be between 0 and 1')
        return round(v, 3)


class SpellCheckItem(BaseModel):
    """
    Represents a spell check item with suggestions.

    Args:
        text (str): The text to spell check.
        type (str): The type of the text.
        offset (int): The offset of the text.
        length (int): The length of the text.
        suggestions (Sequence[SuggestionItem], optional): The list of suggestions for the misspelled text.

    Raises:
        ValueError: If the offset or length is not positive.

    Returns:
        SpellCheckItem: An instance of the SpellCheckItem class.
    """
    text: str
    type: str
    offset: int
    length: int
    suggestions: Sequence[SuggestionItem] = Field(default_factory=list)

    @validator('offset')
    def offset_must_be_positive(cls, v: str) -> str:
        """
        Validates if the offset is positive.

        Args:
            v (int): The offset to validate.

        Raises:
            ValueError: If the offset is not positive.

        Returns:
            int: The validated offset.
        """
        if v < 0:
            raise ValueError('Offset must be positive')
        return v

    @validator('length')
    def length_must_be_positive(cls, v: str) -> str:
        """
        Validates if the length is positive.

        Args:
            v (int): The length to validate.

        Raises:
            ValueError: If the length is not positive.

        Returns:
            int: The validated length.
        """
        if v < 0:
            raise ValueError('Length must be positive')
        return v


class SpellCheckDataClass(BaseModel):
    """
    Represents a spell check model with a list of spell check items.

    Args:
        text (str): The text to spell check.
        items (Sequence[SpellCheckItem], optional): The list of spell check items.

    Returns:
        SpellCheckDataClass: An instance of the SpellCheckDataClass class.
    """
    text: str
    items: Sequence[SpellCheckItem] = Field(default_factory=list)