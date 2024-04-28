import os

import pytest

from edenai_apis import Image
from edenai_apis.features.image.search import SearchDataClass
from edenai_apis.features.image.search.get_image import SearchGetImageDataClass
from edenai_apis.features.image.search.get_images import SearchGetImagesDataClass
from edenai_apis.features.image.search.delete_image.search_delete_image_dataclass import (
    SearchDeleteImageDataClass,
)
from edenai_apis.interface import list_providers
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType, ResponseSuccess

image_search_providers = sorted(list_providers(feature="image", subfeature="search"))


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Skip in opensource package cicd workflow",
)
@pytest.mark.parametrize(("provider"), image_search_providers)
@pytest.mark.xdist_group(name="image_search")
class TestImageSearch:
    def test_upload_image(self, provider):
        # Setup
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="search",
            phase="upload_image",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider, "image", "search", "upload_image", feature_args
        )
        try:
            upload_image_method = Image.search__upload_image(provider)
        except AttributeError:
            raise AttributeError("Could not import upload image phase.")

        # Actions
        upload_output = upload_image_method(**feature_args).model_dump()

        # Assert
        assert (
            upload_output.get("standardized_response").get("status") == "success"
        ), "Upload phase failed"

    def test_get_all_images(self, provider):
        # Setup
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="search",
            phase="get_images",
            provider_name=provider,
        )
        try:
            get_images_method = Image.search__get_images(provider)
        except AttributeError:
            raise AttributeError("Could not import get images phase.")

        # Actions
        get_images_output = get_images_method(**feature_args)
        original_response = get_images_output.original_response
        standardized_response = get_images_output.standardized_response

        # Assert
        assert isinstance(
            get_images_output, ResponseType
        ), f"Expected ResponseType but got {type(get_images_output)}"
        assert isinstance(
            standardized_response, SearchGetImagesDataClass
        ), f"Expected SearchGetImagesDataClass but got {type(standardized_response)}"
        assert original_response is not None

    def test_get_image(self, provider):
        # Setup
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="search",
            phase="get_image",
            provider_name=provider,
        )
        try:
            get_image_method = Image.search__get_image(provider)
        except AttributeError:
            raise AttributeError("Could not import get image phase.")

        # Actions
        get_image_output = get_image_method(**feature_args)
        original_response = get_image_output.original_response
        standardized_response = get_image_output.standardized_response

        # Assert
        assert isinstance(
            get_image_output, ResponseType
        ), f"Expected ResponseType but got {type(get_image_output)}"
        assert isinstance(
            standardized_response, SearchGetImageDataClass
        ), f"Expected SearchGetImageDataClass but got {type(standardized_response)}"
        assert original_response is not None, "Original response should not be empty."

    def test_get_image_does_not_exist(self, provider):
        # Setup : prepare a non-existent image
        invalid_image = "image-not-exist.jpg"
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="search",
            phase="get_image",
            provider_name=provider,
        )
        feature_args["image_name"] = invalid_image
        get_image_method = Image.search__get_image(provider)

        # Action and Assert
        with pytest.raises(ProviderException) as exc:
            api_output = get_image_method(**feature_args)
            assert (
                exc is not None
            ), "ProviderException expected but got an empty Exception."

    def test_launch_similarity_api_call(self, provider):
        # Setup
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="search",
            phase="launch_similarity",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider, "image", "search", "upload_image", feature_args
        )
        try:
            launch_similarity_method = Image.search__launch_similarity(provider)
        except AttributeError:
            raise AttributeError("Could not import launch similarity phase.")

        # Actions
        launch_similarity_output = launch_similarity_method(**feature_args)
        original_response = launch_similarity_output.original_response
        standardized_response = launch_similarity_output.standardized_response

        # Assert
        assert isinstance(
            launch_similarity_output, ResponseType
        ), f"Expected ResponseType but got {type(launch_similarity_output)}"
        assert isinstance(
            standardized_response, SearchDataClass
        ), f"Expected SearchDataClass but got {type(standardized_response)}"
        assert original_response is not None

    def test_launch_similarity_saved_output(self, provider):
        # Setup
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, "image", "search", "launch_similarity"
        )

        # Actions
        standardized = compare_responses(
            feature="image",
            subfeature="search",
            phase="launch_similarity",
            response=saved_output["standardized_response"],
        )

        # Assert
        assert standardized, "The output is not standardized"

    def test_delete_image(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="search",
            phase="delete_image",
            provider_name=provider,
        )
        delete_image = Image.search__delete_image(provider)

        delete_image_output = delete_image(**feature_args)
        standardized_response = delete_image_output.standardized_response

        # Assert
        assert isinstance(
            delete_image_output, ResponseType
        ), f"Expected ResponseType but got {type(delete_image_output)}"
        assert isinstance(
            standardized_response, SearchDeleteImageDataClass
        ), f"Expected SearchGetImagesDataClass but got {type(standardized_response)}"
