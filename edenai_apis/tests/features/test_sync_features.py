"""
Test all synchronous subfeatures (except image search and face regonition) for all providers to check if:
- Valid input data
- Saved output for each provider exists and is well standardized
- providers APIs work and their outputs are well standardized
"""

import importlib

import pytest

from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.tests.conftest import global_features, without_async_and_phase
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.exception import ProviderException

INTERFACE_MODULE = importlib.import_module("edenai_apis.interface_v2")


@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_features(filter=without_async_and_phase)["ungrouped_providers"],
)
class TestSyncProviders:
    @pytest.mark.e2e
    def test_feature_with_invalid_file(self, provider, feature, subfeature):
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

    @pytest.mark.e2e
    def test_feature_saved_output(self, provider, feature, subfeature):
        # Step 1 (Setup) :
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, feature, subfeature
        )

        # Step 2 (Action) :
        if feature != "llm":
            standardized = compare_responses(
                feature, subfeature, saved_output["standardized_response"]
            )
        else:
            standardized = compare_responses(feature, subfeature, saved_output)

        # Step 3 (Assert) :
        assert standardized, "The output is not standardized"
