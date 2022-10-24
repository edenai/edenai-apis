from enum import Enum


class AutomlClassificationProviderName(Enum):
    GOOGLE = "google"
    AMAZON = "amazon"


class TrainingType(Enum):
    SINGLE_LABEL_CLASSIFICATION = "Single label classification"
    MULTIPLE_LABELS_CLASSIFICATION = "Multiple labels classification"
