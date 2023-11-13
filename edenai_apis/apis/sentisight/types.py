import enum

from pydantic import BaseModel


class SentisightBackgroundRemovalParams(BaseModel):
    """Sentisight Background Removal API params

    Sentisight don't have any specific params for background removal
    """


class SentisightPreTrainModel(enum.Enum):
    GENERAL_CLASSIFICATION = "General-classification"
    PLACES_CLASSIFICATION = "Places-classification"
    NSFW_CLASSIFICATION = "NSFW-classification"
    OBJECT_DETECTION = "Object-detection"
    INSTANCE_SEGMENTATION = "Instance-segmentation"
    TEXT_RECOGNITION = "Text-recognition"
    POSE_ESTIMATION = "Pose-estimation"
    BACKGROUND_REMOVAL = "Background-removal"
