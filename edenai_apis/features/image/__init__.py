from .explicit_content import (
    ExplicitContentDataClass,
    ExplicitItem,
    explicit_content_arguments,
)
from .face_detection import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceHairColor,
    FaceItem,
    FaceLandmarks,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
    face_detection_arguments,
)
from .landmark_detection import (
    LandmarkVertice,
    LandmarkLocation,
    LandmarkDetectionDataClass,
    LandmarkItem,
    LandmarkLatLng,
    landmark_detection_arguments,
)
from .logo_detection import (
    LogoDetectionDataClass,
    LogoBoundingPoly,
    LogoVertice,
    LogoItem,
    logo_detection_arguments,
)
from .object_detection import (
    ObjectDetectionDataClass,
    ObjectItem,
    object_detection_arguments,
)
from .anonymization import (
    AnonymizationDataClass,
    anonymization_arguments,
)
from .search import (
    SearchDataClass,
    ImageItem,
    search_delete_image_arguments,
    search_get_image_arguments,
    search_get_images_arguments,
    search_launch_similarity_arguments,
    search_upload_image_arguments,
)
from .generation import (
    GenerationDataClass,
    GeneratedImageDataClass,
    generation_arguments,
)
from .face_compare import (
    FaceCompareDataClass,
    FaceCompareBoundingBox,
    FaceMatch,
    face_compare_arguments,
)
from .automl_classification import (
    AutomlClassificationCreateProject,
    AutomlClassificationUploadImage,
    AutomlClassificationRemoveImage,
    AutomlClassificationPrediction,
)
from .image_interface import ImageInterface
