from pydantic import BaseModel, model_validator


class InputsModel(BaseModel):
    text: str
    image: str
    video: str
    image_url: str
    video_url: str

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
            raise ValueError("At least one input must be provided")
        return self  # type: ignore
