"""
    Test image search subfeature
    The tests are handled differently because the subfeature implements phases
"""
import datetime
import json
import os
from typing import Any

import pytest
from edenai_apis.interface import compute_output, list_providers
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.subfeature_handler import call_subfeature

FEATURE = "image"
SUBFEATURE = "search"


def get_providers_image_search():
    """Generate two lists of parameters for tests classes (here for image search).
    Returns:
        dict : two lists
         grouped_providers   : [([provider1, provider2], image, search)]
         ungrouped_providers : [(provider1, imaeg, search),
                                (provider2, imaeg, search)]
    """
    providers = list_providers(FEATURE, SUBFEATURE)
    detailed_providers_list = []
    params_dict = {}
    for provider in providers:
        detailed_params = pytest.param(
            provider,
            FEATURE,
            SUBFEATURE,
            marks=[
                getattr(pytest.mark, provider),
                getattr(pytest.mark, FEATURE),
                getattr(pytest.mark, SUBFEATURE),
            ],
        )
        detailed_providers_list.append(detailed_params)

    grouped_providers_list = [
        pytest.param(
            providers,
            feature,
            subfeature,
            marks=[getattr(pytest.mark, feature), getattr(pytest.mark, subfeature)],
        )
        for ((feature, subfeature), providers) in params_dict.items()
    ]
    return {
        "grouped_providers": grouped_providers_list,
        "ungrouped_providers": detailed_providers_list,
    }


class CommonImageSearchTests:
    """All tests needed to test provider/ image -- search for all its phases
    (upload--delete--get list images--get image-- + similarity results)
    """

    def _test_upload_(self, provider):
        # Test image upload, if the project fails
        """Test API Call for the upload phase"""
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            phase="upload_image",
        )
        status = call_subfeature(
            provider_name=provider,
            subfeature=SUBFEATURE,
            feature=FEATURE,
            phase="upload_image",
            args=args,
        )
        print("status :", status)
        assert status["status"] == "success"

    def _test_delete(self, provider):
        """Test API Call for the delete phase"""
        print("Deleting image...")
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            phase="delete_image",
        )
        status = call_subfeature(
            provider_name=provider,
            subfeature=SUBFEATURE,
            feature=FEATURE,
            phase="delete_image",
            args=args,
        )
        print("status :", status)
        assert status["status"] == "success"

    def _test_get_list_images_(self, provider):
        """Test API Call for the get list images phases"""
        print("Getting list images...")
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            phase="get_images",
        )

        result = call_subfeature(
            provider_name=provider,
            subfeature=SUBFEATURE,
            feature=FEATURE,
            phase="get_images",
            args=args,
        ).dict()

        assert isinstance(result, dict)
        original_response = result.get("original_response")
        standarized_response = result.get("standarized_response")
        assert original_response is not None
        assert standarized_response is not None
        assert isinstance(standarized_response["list_images"], list)

    def _test_get_image_(self, provider):
        print("Getting image...")
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            phase="get_image",
        )
        image = call_subfeature(
            provider_name=provider,
            subfeature=SUBFEATURE,
            feature=FEATURE,
            phase="get_image",
            args=args,
        ).dict()

        assert isinstance(image, dict)
        original_response = image.get("original_response")
        standarized_response = image.get("standarized_response")
        assert original_response is not None
        assert standarized_response is not None
        assert isinstance(standarized_response["image"], str)

    def _test_real_output_search_(self, provider):
        """Test API Call for launch search similarity"""
        print("Launch search...")
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            phase="launch_similarity",
        )
        # saved_output = load_output(provider, FEATURE, SUBFEATURE)
        api_output_search = call_subfeature(
            provider_name=provider,
            subfeature=SUBFEATURE,
            feature=FEATURE,
            phase="launch_similarity",
            args=args,
        ).dict()
        assert isinstance(api_output_search, dict)

        original_response = api_output_search.get("original_response")
        standarized_response = api_output_search.get("standarized_response")
        assert original_response is not None
        assert standarized_response is not None

        # compare provider output with standard response to verify standarization
        assert compare_responses(
            FEATURE, SUBFEATURE, standarized_response, phase="launch_similarity"
        )

        # test output is JSON serializable
        def default(output: Any):
            if isinstance(output, (datetime.date, datetime.datetime)):
                return output.isoformat()
            return None

        assert json.dumps(api_output_search["original_response"], default=default)
        assert json.dumps(api_output_search["standarized_response"], default=default)

    def _test_saved_output_search(self, provider):
        """Test API Call"""
        # We test stored provider result vs subfeature general result
        # Should fail tests
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, FEATURE, SUBFEATURE, "launch_similarity"
        )
        assert compare_responses(
            FEATURE,
            SUBFEATURE,
            saved_output["standarized_response"],
            phase="launch_similarity",
        )

    def _test_compute_image_search_output(self, provider):
        """Test can call compute subfeature with fake = True"""
        # This should always work
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=FEATURE,
            subfeature=SUBFEATURE,
            phase="launch_similarity",
        )
        final_result = compute_output(
            provider,
            FEATURE,
            SUBFEATURE,
            args,
            phase="launch_similarity",
            fake=True,
        )
        assert final_result
        assert final_result["status"]


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD", reason="Don't run this on CI/CD"
)
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    get_providers_image_search()["ungrouped_providers"],
)
class TestProviderImageSearch(CommonImageSearchTests):
    def test_search_phases_image(self, provider, feature, subfeature):
        self._test_upload_(provider)

        self._test_get_image_(provider)

        self._test_get_list_images_(provider)

        self._test_delete(provider)

    def test_image_search_output(self, provider, feature, subfeature):
        self._test_real_output_search_(provider)
        self._test_saved_output_search(provider)

    def test_compute_image_search_output(self, provider, feature, subfeature):
        """Test call compute subfeature with fake = True"""
        self._test_compute_image_search_output(provider)


pytest.mark.skipif(os.environ.get("TEST_SCOPE") != "CICD", reason="Run On CICD")


@pytest.mark.parametrize(
    ("providers", "feature", "subfeature"),
    get_providers_image_search()["grouped_providers"],
)
class TestFeatureSubfeatureImageSearch(CommonImageSearchTests):
    """Test all providers  for Image/Search.
    Tests only fail if all providers fail or one subfeature.
    """

    def test_subfeature_output(self, providers, feature, subfeature):
        """Test API Call"""
        failures = []
        for provider in providers:
            self._test_saved_output_search(provider)
            try:
                self._test_real_output_search_(provider)
            except Exception as exc:
                failures.append(exc)

        print(failures)
        assert len(providers) != len(failures)


    def test_compute_image_search_output(self, providers, feature, subfeature):
        """Test call compute subfeature with fake = True"""
        for provider in providers:
            self._test_compute_image_search_output(provider)
