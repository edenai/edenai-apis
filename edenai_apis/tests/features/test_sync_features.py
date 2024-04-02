"""
    Test all synchronous subfeatures (except image search and face regonition) for all providers to check if:
    - Valid input data
    - Saved output for each provider exists and is well standardized
    - providers APIs work and their outputs are well standardized
"""
import importlib
import os
import traceback

import pytest

from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.tests.conftest import global_features, without_async_and_phase
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.conversion import iterate_all
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType

from edenai_apis.interface import IS_MONITORING
from edenai_apis.utils.monitoring import insert_api_call

INTERFACE_MODULE = importlib.import_module("edenai_apis.interface_v2")


class CommonTestsSubfeatures:
    def _test_feature_with_invalid_file(self, provider, feature, subfeature):
        # Setup
        invalid_file = "fakefile.txt"
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS, feature=feature, subfeature=subfeature
        )
        feature_validated_args = validate_all_provider_constraints(
            provider, feature, subfeature, "", feature_args
        )
        if not feature_validated_args.get("file"):
            pytest.skip("unsupported configuration")
        feature_validated_args["file"] = invalid_file

        # Action
        with pytest.raises(
            (ProviderException, AttributeError, FileNotFoundError)
        ) as exc:
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_method = getattr(feature_class, f"{subfeature}")(provider)
            provider_method(**feature_validated_args)
            assert (
                exc is not None
            ), "ProviderException, AttributeError or FileNotFoundError expected."

    def _test_feature_api_call(self, provider, feature, subfeature):
        # Setup
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS, feature=feature, subfeature=subfeature
        )
        validated_args = validate_all_provider_constraints(
            provider, feature, subfeature, "", feature_args
        )
        try:
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_method = getattr(feature_class, f"{subfeature}")(provider)
        except AttributeError:
            raise ("Could not import provider method.")

        # Actions
        provider_api_output = provider_method(**validated_args)
        provider_api_dict = provider_api_output.model_dump()
        original_response = provider_api_dict.get("original_response")
        standardized_response = provider_api_dict.get("standardized_response")
        standardized = compare_responses(feature, subfeature, standardized_response)

        # Step 3 (asserts) : check dataclass standardization
        assert isinstance(
            provider_api_output, ResponseType
        ), f"Expected ResponseType but got {type(provider_api_output)}"
        assert original_response is not None, "original_response should not be None"
        assert (
            standardized_response is not None
        ), "standardized_response should not be None"
        assert standardized, "The output is not standardized"

        assert any(
            [std != None for std in iterate_all(standardized_response, "value")]
        ), "Response shouldn't be empty"

        if IS_MONITORING:
            insert_api_call(provider, feature, subfeature, None, None)

    def _test_feature_saved_output(self, provider, feature, subfeature):
        # Step 1 (Setup) :
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, feature, subfeature
        )

        # Step 2 (Action) :
        standardized = compare_responses(
            feature, subfeature, saved_output["standardized_response"]
        )

        # Step 3 (Assert) :
        assert standardized, "The output is not standardized"


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD", reason="Don't run this on CI/CD"
)
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_features(filter=without_async_and_phase)["ungrouped_providers"],
)
class TestSyncProviders(CommonTestsSubfeatures):
    @pytest.mark.skipif(
        os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
        reason="Skip in opensource package cicd workflow",
    )
    def test_feature_with_invalid_file(self, provider, feature, subfeature):
        self._test_feature_with_invalid_file(provider, feature, subfeature)

    @pytest.mark.skipif(
        os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
        reason="Skip in opensource package cicd workflow",
    )
    def test_feature_api_call(self, provider, feature, subfeature):
        self._test_feature_api_call(provider, feature, subfeature)

    def test_feature_saved_output(self, provider, feature, subfeature):
        self._test_feature_saved_output(provider, feature, subfeature)


@pytest.mark.skipif(os.environ.get("TEST_SCOPE") != "CICD", reason="Run On CICD")
@pytest.mark.parametrize(
    ("providers", "feature", "subfeature"),
    global_features(filter=without_async_and_phase)["grouped_providers"],
)
class TestSyncSubfeatures(CommonTestsSubfeatures):
    def test_sync_subfeature(self, providers, feature, subfeature):
        """Test API Call"""
        failures = []
        # List of providers with wokring / not working
        for provider in providers:
            self._test_feature_saved_output(provider, feature, subfeature)
            try:
                self._test_feature_api_call(provider, feature, subfeature)
                self._test_feature_with_invalid_file(provider, feature, subfeature)
            except Exception as exc:
                print(traceback.format_exc())
                failures.append(exc)

        # Only fail if all providers failes in a certain feature/subfeature
        print(failures)
        if "missing 1 required positional argument: 'model'" in str(failures):
            pytest.skip("we skip test because of missing model argument. This should be tested in implementation of the feature")
        assert len(providers) != len(failures)

    def test_sync_subfeature_fake(self, providers, feature, subfeature):
        """Fake call"""
        for provider in providers:
            self._test_feature_saved_output(provider, feature, subfeature)
