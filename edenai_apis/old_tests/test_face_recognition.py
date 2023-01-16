from datetime import datetime
import os

import pytest
from edenai_apis import Image
from edenai_apis.features.image.face_detection.face_detection_args import \
    face_detection_arguments
from edenai_apis.features.image.face_recognition.create_collection.face_recognition_create_collection_dataclass import \
    FaceRecognitionCreateCollectionDataClass
from edenai_apis.features.image.face_recognition.recognize.face_recognition_recognize_args import (
    face_recognition_recognize_arguments, get_data_files)
from edenai_apis.features.image.face_recognition.list_faces.face_recognition_list_faces_dataclass import \
    FaceRecognitionListFacesDataClass
from edenai_apis.features.image.landmark_detection.landmark_detection_args import \
    landmark_detection_arguments
from edenai_apis.interface import list_providers
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType

# So that we don't have duplicate collections if test fails
# don't use uuid as it will fail pytest multiprocess tests
# 
collection_id = f"test_{datetime.now().strftime('%y%m%d%H%M')}"
test_params = sorted(list(
    map(
        lambda provider: (provider, collection_id),
        list_providers(feature="image", subfeature="face_recognition"),
    )
))

@pytest.mark.skipif(os.environ.get("TEST_SCOPE") == 'CICD-OPENSOURCE', reason="Don't run on opensource cicd workflow")
@pytest.mark.xdist_group(name="face_recognition")
@pytest.mark.parametrize(("provider", "collection_id"), test_params)
class TestFaceRecognition:
    def test_create_collection(self, provider, collection_id):
        """Successfully create a collection"""
        create_collection = Image.face_recognition__create_collection(provider)
        response = create_collection(collection_id=collection_id)
        assert isinstance(response, FaceRecognitionCreateCollectionDataClass)
        assert isinstance(response.collection_id, str)

    def test_list_collections(self, provider, collection_id):
        lsit_collections = Image.face_recognition__list_collections(provider)
        response = lsit_collections()
        collections = response.standardized_response.collections
        assert len(collections) > 0
        assert collection_id in collections

    def test_create_collection_already_exists(self, provider, collection_id):
        """creating a collection that already exists should raise an error"""
        with pytest.raises(ProviderException) as exc:
            create_collection = Image.face_recognition__create_collection(provider)
            create_collection(collection_id=collection_id)
            assert exc is not None

    def test_list_faces(self, provider, collection_id):
        list_faces = Image.face_recognition__list_faces(provider)
        response = list_faces(collection_id=collection_id)
        assert isinstance(response, ResponseType)
        std_response = response.standardized_response
        assert isinstance(std_response, FaceRecognitionListFacesDataClass)
        assert std_response.face_ids == []

    def test_list_faces_does_not_exists(self, provider, collection_id):
        with pytest.raises(ProviderException) as exc:
            list_faces = Image.face_recognition__list_faces(provider)
            list_faces(collection_id="test_does_not_exists")
            assert exc is not None

    def test_add_face_to_collection(self, provider, collection_id):
        list_faces = Image.face_recognition__list_faces(provider)
        collection = list_faces(collection_id=collection_id)
        assert len(collection.standardized_response.face_ids) == 0

        img = face_recognition_recognize_arguments()["file"]
        add_face = Image.face_recognition__add_face(provider)
        response = add_face(collection_id=collection_id, file=img)
        face_ids = response.standardized_response.face_ids

        collection = list_faces(collection_id=collection_id)
        assert sorted(collection.standardized_response.face_ids) == sorted(face_ids)

    def test_add_face_to_collection_no_face(self, provider, collection_id):
        img = landmark_detection_arguments()["file"]
        add_face = Image.face_recognition__add_face(provider)
        with pytest.raises(ProviderException) as exc:
            add_face(collection_id=collection_id, file=img)
            assert exc is not None

    def test_delete_face_from_collection(self, provider, collection_id):
        list_faces = Image.face_recognition__list_faces(provider)
        collection = list_faces(collection_id=collection_id)
        face_ids = collection.standardized_response.face_ids
        collection_length = len(face_ids)
        assert collection_length > 0

        face_to_delete = face_ids[0]
        delete_face = Image.face_recognition__delete_face(provider)
        response = delete_face(collection_id=collection_id, face_id=face_to_delete)
        assert response.standardized_response.deleted is True

        list_faces = Image.face_recognition__list_faces(provider)
        collection = list_faces(collection_id=collection_id)
        updated_face_ids = collection.standardized_response.face_ids
        assert face_ids != updated_face_ids
        assert face_to_delete not in updated_face_ids

    def test_delete_face_from_collection_wrong_id(self, provider, collection_id):
        with pytest.raises(ProviderException) as exc:
            delete_face = Image.face_recognition__delete_face(provider)
            delete_face(collection_id=collection_id, face_id="test_does_not_exists")
            assert exc is not None

    def test_recognize(self, provider, collection_id):
        add_face = Image.face_recognition__add_face(provider)
        images = get_data_files()  # contains images with face of the same person
        for image in images[:-1]:
            add_face(collection_id=collection_id, file=image)
        # image with face different person
        other_image = face_detection_arguments()["file"]
        add_face_response = add_face(collection_id=collection_id, file=other_image)
        face_id = add_face_response.standardized_response.face_ids[0]

        recognize = Image.face_recognition__recognize(provider)
        result = recognize(collection_id=collection_id, file=images[-1])
        std_response = result.standardized_response
        assert len(std_response.items) > 0

        assert face_id not in [face.face_id for face in std_response.items]

    def test_recognize_no_face(self, provider, collection_id):
        img = landmark_detection_arguments()["file"]
        recognize = Image.face_recognition__recognize(provider)
        with pytest.raises(ProviderException) as exc:
            recognize(collection_id=collection_id, file=img)
            assert exc is not None

    def test_recognize_empty_collection(self, provider, collection_id):
        list_faces = Image.face_recognition__list_faces(provider)
        response = list_faces(collection_id=collection_id)
        face_ids = response.standardized_response.face_ids

        image = face_recognition_recognize_arguments()["file"]
        delete_face = Image.face_recognition__delete_face(provider)
        for face_id in face_ids:
            delete_face(collection_id, face_id)

        recognize = Image.face_recognition__recognize(provider)
        with pytest.raises(ProviderException) as exc:
            recognize(collection_id=collection_id, file=image)
            assert exc is not None

    def test_delete_collection(self, provider, collection_id):
        delete_collection = Image.face_recognition__delete_collection(provider)
        response = delete_collection(collection_id=collection_id)
        assert response.standardized_response.deleted is True

    def test_delete_collection_does_not_exists(self, provider, collection_id):
        with pytest.raises(ProviderException) as exc:
            delete_collection = Image.face_recognition__delete_collection(provider)
            delete_collection(collection_id="test_does_not_exists")
            assert exc is not None



            
