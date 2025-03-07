"""
Test all async subfeatures for all providers to check if:
- Job ID is not null
- Saved output for each provider exists and is well standardized
- providers APIs work and their outputs are well standardized
"""

import importlib
import logging

import pytest

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.tests.conftest import global_features, only_async_without_phase
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.exception import AsyncJobExceptionReason, ProviderException

MAX_TIME = 300
TIME_BETWEEN_CHECK = 10
INTERFACE_MODULE = importlib.import_module("edenai_apis.interface_v2")


@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_features(only_async_without_phase)["ungrouped_providers"],
)
class TestAsyncSubFeatures:

    @pytest.mark.e2e
    def test_launch_job_invalid_parameters(self, provider, feature, subfeature):
        logging.info(
            f"Testing launch job with invalid parameters for {provider}, {subfeature}..\n"
        )

        # Step 1 (setup) : prepare invalid parameters
        invalid_feature_args = {"invalid_key": "invalid_value"}
        try:
            subfeature_suffix = "__launch_job"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_launch_job_function = getattr(
                feature_class, f"{subfeature}{subfeature_suffix}"
            )(provider)
        except AttributeError:
            raise AttributeError("Could not import provider launch job method.")

        # Step 2 & 3 (action & assert ) : Call the feature with invalid parameters
        with pytest.raises(TypeError):
            provider_launch_job_function(**invalid_feature_args)

    @pytest.mark.e2e
    def test_get_job_result_does_not_exist(self, provider, feature, subfeature):
        logging.info(
            f"Testing get job result with invalid id for {provider}, {subfeature}..\n"
        )
        # Step 1 (setup) : prepare a non-existent job ID

        job_id = "12345678-1234-1234-1234-123456789abc"
        custom_job_id = "12345678-1234-1234-1234-123456789abcEdenAIabdkla32421221akdakj"
        google_id_operation = "projects/148983085864/locations/europe-west1/operations/5890021581918958336"

        exception_message_error = "Should return the right exception indicating that the job id is either wrong or old"

        try:
            subfeature_suffix = "__get_job_result"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_get_job_result_method = getattr(
                feature_class, f"{subfeature}{subfeature_suffix}"
            )(provider)
        except AttributeError:
            raise AttributeError("Could not import provider get job method.")

        # Step 2 & 3 (action & assert) : Call the feature with the non-existent job ID
        try:
            provider_get_job_result_method(job_id)
        except (
            ValueError
        ) as value_error_excp:  # job id is composed of more than one information
            try:
                provider_get_job_result_method(custom_job_id)
            except ProviderException as prov_excp:
                assert AsyncJobExceptionReason.DEPRECATED_JOB_ID.value in str(
                    prov_excp
                ), exception_message_error
        except (
            ProviderException
        ) as provider_excp:  # google provider job id has sometimes has another structure
            if "does not match the pattern" in str(provider_excp):
                try:
                    provider_get_job_result_method(google_id_operation)
                except ProviderException as prov_excp:
                    assert AsyncJobExceptionReason.DEPRECATED_JOB_ID.value in str(
                        prov_excp
                    ), exception_message_error
            else:
                # str(provider_excp) == Provider returned an empty response is necessary for openai which return an empty response
                assert (
                    AsyncJobExceptionReason.DEPRECATED_JOB_ID.value
                    in str(provider_excp)
                    or str(provider_excp) == "Provider returned an empty response"
                ), exception_message_error

    @pytest.mark.unit
    def test_api_get_job_result_saved_output(self, provider, feature, subfeature):
        logging.info(
            f"Testing get job result with saved output for {provider}, {subfeature}..\n"
        )

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
