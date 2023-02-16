from abc import abstractmethod
from io import BufferedReader
from typing import List

from edenai_apis.features.image.anonymization.anonymization_dataclass import (
    AnonymizationDataClass,
)
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceDetectionDataClass,
)
from edenai_apis.features.image.face_recognition.add_face.face_recognition_add_face_dataclass import (
    FaceRecognitionAddFaceDataClass,
)
from edenai_apis.features.image.face_recognition.create_collection.face_recognition_create_collection_dataclass import (
    FaceRecognitionCreateCollectionDataClass,
)
from edenai_apis.features.image.face_recognition.delete_collection.face_recognition_delete_collection_dataclass import FaceRecognitionDeleteCollectionDataClass
from edenai_apis.features.image.face_recognition.delete_face.face_recognition_delete_face_dataclass import FaceRecognitionDeleteFaceDataClass
from edenai_apis.features.image.face_recognition.recognize.face_recognition_recognize_dataclass import (
    FaceRecognitionRecognizeDataClass,
)
from edenai_apis.features.image.face_recognition.list_collections.face_recognition_list_collections_dataclass import (
    FaceRecognitionListCollectionsDataClass,
)
from edenai_apis.features.image.face_recognition.list_faces.face_recognition_list_faces_dataclass import (
    FaceRecognitionListFacesDataClass,
)
from edenai_apis.features.image.landmark_detection.landmark_detection_dataclass import (
    LandmarkDetectionDataClass,
)
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoDetectionDataClass,
)
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectDetectionDataClass,
)
from edenai_apis.features.image.search.delete_image.search_delete_image_dataclass import (
    SearchDeleteImageDataClass,
)
from edenai_apis.features.image.search.get_image.search_get_image_dataclass import (
    SearchGetImageDataClass,
)
from edenai_apis.features.image.search.get_images.search_get_images_dataclass import (
    SearchGetImagesDataClass,
)
from edenai_apis.features.image.search.search_dataclass import SearchDataClass
from edenai_apis.features.image.search.upload_image.search_upload_image_dataclass import (
    SearchUploadImageDataClass,
)
from edenai_apis.features.image.generation.generation_dataclass import (
    GenerationDataClass
)
from edenai_apis.utils.types import ResponseType


class ImageInterface:
    @abstractmethod
    def image__anonymization(
        self, file: BufferedReader
    ) -> ResponseType[AnonymizationDataClass]:
        """
        Anonymize face, names, car plates etc from an image

        Args:
            file (BufferedReader): image to anonymize
        """
        raise NotImplementedError

    @abstractmethod
    def image__explicit_content(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ExplicitContentDataClass]:
        """
        Detect explicit content in an image

        Args:
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[FaceDetectionDataClass]:
        """
        Detect faces in an image

        Args:
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__landmark_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[LandmarkDetectionDataClass]:
        """
        Detect popular landmark in an image
        (eg: popular tourist spots like Eiffel Tower)

        Args:
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__logo_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[LogoDetectionDataClass]:
        """
        Detect Logo in an image

        Args:
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__object_detection(
        self, 
        file: str,
        file_url: str= ""
    ) -> ResponseType[ObjectDetectionDataClass]:
        """
        Detect objects in an image

        Args:
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__search__create_project(self, project_name: str) -> str:
        """
        Create an image search project

        Args:
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    ## TO DO better response types for image search
    @abstractmethod
    def image__search__upload_image(
        self, file: BufferedReader, image_name: str, project_id: str
    ) -> SearchUploadImageDataClass:
        """
        Upload image for an image search project

        Args:
            file (BufferedReader): image to analyze
            image_name (str): name of the image
            project_id (str): image search project id
        """
        raise NotImplementedError

    @abstractmethod
    def image__search__delete_image(
        self, image_name: str, project_id: str
    ) -> SearchDeleteImageDataClass:
        """
        Delete image of an image search project

        Args:
            image_name (str): name of the image
            project_id (str): image search project id
        """
        raise NotImplementedError

    @abstractmethod
    def image__search__get_images(
        self, project_id: str
    ) -> ResponseType[SearchGetImagesDataClass]:
        """
        get images of an image search project

        Args:
            project_id (str): image search project id
        """
        raise NotImplementedError

    @abstractmethod
    def image__search__get_image(
        self, image_name: str, project_id: str
    ) -> ResponseType[SearchGetImageDataClass]:
        """
        get image of an image search project

        Args:
            image_name (str): name of the image
            project_id (str): image search project id
        """
        raise NotImplementedError

    @abstractmethod
    def image__search__launch_similarity(
        self, file: BufferedReader, project_id: str
    ) -> ResponseType[SearchDataClass]:
        """
        Launch similarity analysis of a search image project

        Args:
            file (BufferedReader): image to analyze
            project_id (str): image search project id
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__create_collection(
        self, collection_id: str
    ) -> FaceRecognitionCreateCollectionDataClass:
        """
        Create a Face Collection to analyze for face recognition

        Args:
            collection_id (str): ID given to created collection
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__list_collections(
        self,
    ) -> ResponseType[FaceRecognitionListCollectionsDataClass]:
        """
        List all Face Collections
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__list_faces(
        self, collection_id: str
    ) -> ResponseType[FaceRecognitionListFacesDataClass]:
        """
        List a Faces of a Collection

        Args:
            collection_id (str): ID of collection
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__delete_collection(self, collection_id: str) -> ResponseType[FaceRecognitionDeleteCollectionDataClass]:
        """
        Delete a Face Collection

        Args:
            collection_id (str): ID of collection
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__add_face(
        self, collection_id: str, file: BufferedReader
    ) -> ResponseType[FaceRecognitionAddFaceDataClass]:
        """
        Detect and add a face to a collection from an image

        Args:
            collection_id (str): ID of collection
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__delete_face(self, collection_id, face_id) -> ResponseType[FaceRecognitionDeleteFaceDataClass]:
        """
        Delete a face from collection

        Args:
            collection_id (str): ID of collection
            face_id (str): ID of face to delete
        """
        raise NotImplementedError

    @abstractmethod
    def image__face_recognition__recognize(
        self, collection_id: str, file: BufferedReader
    ) -> ResponseType[FaceRecognitionRecognizeDataClass]:
        """
        Detect the biggers face from image and try
        to find faces from the same person in the face collection

        Args:
            collection_id (str): ID of collection
            file (BufferedReader): image to analyze
        """
        raise NotImplementedError

    @abstractmethod
    def image__generation(
        self, text: str, num_images : int = 1, width : int = 512, height : int = 512
    ) -> ResponseType[GenerationDataClass]:
        """
        Generate an image based on a text prompt.

        Args:
            text(str): prompt of the image to generate
        """
        raise NotImplementedError
