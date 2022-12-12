"""
    Test all subfeatures (except image search) for all providers to check if:
    - Saved output for each provider exists and is well standardized
    - providers APIs work and their outputs are well standardized
"""
import datetime
import json
import traceback
import os
import pytest
from edenai_apis.interface import compute_output, list_features
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider

from edenai_apis.tests.utils.outputs import test_outputs
from edenai_apis.utils.compare import compare, compare_responses


def global_method_list_as_params():
    """Generate two lists of parameters for tests classes.
    Returns:
        dict : two lists
         grouped_providers   : [([provider1, provider2], feature, subfeature)]
         ungrouped_providers : [(provider1, feature, subfeature),
                                (provider2, feature, subfeature)]
    """
    method_list = list_features()
    params_dict = {}
    detailed_providers_list = []

    for provider, feature, subfeature, *phase in method_list:
        if not phase:
            # test available in test_image_search
            if feature == "image" and subfeature == "search":
                continue

            detailed_params = pytest.param(
                provider,
                feature,
                subfeature,
                marks=[
                    getattr(pytest.mark, provider),
                    getattr(pytest.mark, feature),
                    getattr(pytest.mark, subfeature),
                ],
            )
            detailed_providers_list.append(detailed_params)

            if (feature, subfeature) in params_dict.keys():
                params_dict[(feature, subfeature)].append(provider)
            else:
                params_dict[(feature, subfeature)] = [provider]

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


class CommonProvidersSubfeaturesTests:
    """All tests needed to test provider/feature/subfeature"""

    def _test_saved_output(self, provider, feature, subfeature, phase=""):
        """Test API Call"""
        # We test stored provider result vs subfeature general result
        # Should fail tests
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, feature, subfeature, phase
        )
        assert compare_responses(
            feature, subfeature, saved_output["standardized_response"], phase=phase
        )

    def _test_real_output(self, provider, feature, subfeature, phase=""):
        # get provider output with API call

        # skip in opensource package cicd workflow
        if os.environ.get("TEST_SCOPE") == 'CICD-OPENSOURCE':
            return
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, feature, subfeature, phase
        )
        api_output = test_outputs(provider, feature, subfeature, phase, generate=False)
        if not isinstance(api_output, dict):
            api_output = api_output.dict()
        assert isinstance(api_output, dict)
        original_response = api_output.get("original_response")
        standardized_response = api_output.get("standardized_response")
        assert original_response is not None
        assert standardized_response is not None

        # compare provider output with standard response to verify standarization
        assert compare_responses(feature, subfeature, standardized_response, phase=phase)

        print("compare standardized output")
        assert compare(standardized_response, saved_output["standardized_response"])

        # test output is JSON serializable
        def default(output):
            if isinstance(output, (datetime.date, datetime.datetime)):
                return output.isoformat()
            return None

        assert json.dumps(api_output["original_response"], default=default)
        assert json.dumps(api_output["standardized_response"], default=default)

    def _test_compute_subfeature_output(self, provider, feature, subfeature, phase=""):
        """Test can call compute subfeature with fake = True"""
        # This should always work
        args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
            phase=phase,
        )
        final_result = compute_output(
            provider, feature, subfeature, args, phase, fake=True
        )
        assert final_result
        assert final_result["status"]


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD", reason="Don't run this on CI/CD"
)
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_method_list_as_params()["ungrouped_providers"],
)
class TestProviderFeatureSubfeature(CommonProvidersSubfeaturesTests):
    """All test for one tripple provider/feature/subfeature.
    Tests fail if one subfeature of one provider doesn't work.
    """

    def test_subfeature_output(self, provider, feature, subfeature):
        """Test API Call"""
        print("THE FEATURE IS ", feature)
        print("THE SUBFEATURE IS", subfeature)
        self._test_saved_output(provider, feature, subfeature)
        self._test_real_output(provider, feature, subfeature)

    def test_compute_subfeature_output(self, provider, feature, subfeature):
        """Test call compute subfeature with fake = True"""
        self._test_compute_subfeature_output(provider, feature, subfeature)


@pytest.mark.skipif(os.environ.get("TEST_SCOPE") != "CICD", reason="Run On CICD")
@pytest.mark.parametrize(
    ("providers", "feature", "subfeature"),
    global_method_list_as_params()["grouped_providers"],
)
class TestFeatureSubfeature(CommonProvidersSubfeaturesTests):
    """Test all providers  for one feature/subfeature.
    Tests only fail if all providers fail for a subfeature.
    """

    def test_subfeature_output(self, providers, feature, subfeature):
        """Test API Call"""
        failures = []
        # List of providers with wokring / not working
        for provider in providers:
            self._test_saved_output(provider, feature, subfeature)
            try:
                self._test_real_output(provider, feature, subfeature)
            except Exception as exc:
                print(traceback.format_exc())
                failures.append(exc)

        # Only fail if all providers failes in a certain feature/subfeature
        print(failures)
        assert len(providers) != len(failures)


    def test_compute_subfeature_output(self, providers, feature, subfeature):
        """Test call compute subfeature with fake = True"""
        for provider in providers:
            self._test_compute_subfeature_output(provider, feature, subfeature)
