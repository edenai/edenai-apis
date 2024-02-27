from typing import Optional

from pydantic import BaseModel, model_validator, ValidationError


class InputsModel(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None

    # HACK: Some problems with MyPy are ignored here
    # because of the incompatibility between Pydantic and MyPy.
    @model_validator(mode="after")  # type: ignore
    def check_all_inputs(self) -> "InputsModel":
        if not any(
            [
                self.text,
                self.image,
                self.video,
                self.image_url,
                self.video_url,
            ]
        ):
            raise ValidationError("At least one input must be provided")
        return self  # type: ignore
