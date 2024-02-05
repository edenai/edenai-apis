"""
    Test all async subfeatures for all providers to check if:
    - Job ID is not null
    - Saved output for each provider exists and is well standardized
    - providers APIs work and their outputs are well standardized
"""
import importlib
import logging
import os
import traceback
from time import sleep

import pytest

from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.tests.conftest import global_features, only_async_without_phase
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.conversion import iterate_all
from edenai_apis.utils.exception import AsyncJobExceptionReason, ProviderException
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType
from edenai_apis.utils.compare import compare_responses
from edenai_apis.interface import IS_MONITORING
from edenai_apis.utils.monitoring import insert_api_call

MAX_TIME = 180
TIME_BETWEEN_CHECK = 10
INTERFACE_MODULE = importlib.import_module("edenai_apis.interface_v2")


class CommonAsyncTests:
    def _test_launch_job_id(self, provider, feature, subfeature):
        logging.info(f"Testing launch job id for {provider}, {subfeature}..\n")

        # Step 1 (setup) : prepare parameters
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS, feature=feature, subfeature=subfeature
        )
        validated_args = validate_all_provider_constraints(
            provider, feature, subfeature, "", feature_args
        )
        try:
            subfeature_suffix = "__launch_job"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_launch_job_function = getattr(
                feature_class, f"{subfeature}{subfeature_suffix}"
            )(provider)
        except AttributeError:
            raise AttributeError("Could not import provider launch job method.")

        # Step 2 (action) : Launch the job (action)
        launch_job_response: AsyncLaunchJobResponseType = provider_launch_job_function(
            **validated_args
        )

        # Step 3 (assert) : Assert job_id
        assert (
            launch_job_response.provider_job_id is not None
        ), "provider job id should not be null."

        pytest.job_id = launch_job_response.provider_job_id

    def _test_api_get_job_result_real_output(self, provider, feature, subfeature):
        logging.info(
            f"Testing get job result with real output for {provider}, {subfeature}..\n"
        )
        # skip in opensource package cicd workflow
        if os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE":
            return

        # Step 1 (setup) : prepare parameters
        provider_job_id = pytest.job_id
        try:
            subfeature_suffix = "__get_job_result"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_get_job_result_method = getattr(
                feature_class, f"{subfeature}{subfeature_suffix}"
            )(provider)
        except AttributeError:
            raise AttributeError("Could not import provider get job method.")

        # Step 2 (actions) : call get job result with a valid job id
        sleep(5)
        current_time = MAX_TIME
        while current_time > 0:
            print(f"wait job result {MAX_TIME- current_time}s")
            provider_api_output = provider_get_job_result_method(provider_job_id)
            provider_api_output_dict = provider_api_output.model_dump()
            if provider_api_output_dict["status"] != "pending":
                provider_api_output = provider_api_output
                break
            current_time = current_time - TIME_BETWEEN_CHECK
            sleep(TIME_BETWEEN_CHECK)

        original_response = provider_api_output_dict.get("original_response")
        standardized_response = provider_api_output_dict.get("standardized_response")
        standardized = compare_responses(feature, subfeature, standardized_response)

        # Step 3 (asserts) : check dataclass standardization
        assert isinstance(
            provider_api_output, AsyncBaseResponseType
        ), f"Expected AsyncBaseResponseType but got {type(provider_api_output)}"
        assert original_response is not None, "original_response should not be None"
        assert (
            standardized_response is not None
        ), "standardized_response should not be None"
        assert any(
            [std != None for std in iterate_all(standardized_response, "value")]
        ), "Response shouldn't be empty"
        assert standardized, "The output is not standardized"

        if IS_MONITORING:
            insert_api_call(provider, feature, subfeature, None, None)

    def _test_async_job(self, provider, feature, subfeature):
        self._test_launch_job_id(provider, feature, subfeature)
        self._test_api_get_job_result_real_output(provider, feature, subfeature)

    def _test_launch_job_invalid_parameters(self, provider, feature, subfeature):
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

    def _test_get_job_result_does_not_exist(self, provider, feature, subfeature):
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

    def _test_api_get_job_result_saved_output(self, provider, feature, subfeature):
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


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD", reason="Don't run this on CI/CD"
)
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_features(only_async_without_phase)["ungrouped_providers"],
)
class TestAsyncSubFeatures(CommonAsyncTests):
    @pytest.mark.skipif(
        os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
        reason="Skip in opensource package cicd workflow",
    )
    def test_async_job(self, provider, feature, subfeature):
        self._test_async_job(provider, feature, subfeature)

    def test_api_get_job_result_saved_output(self, provider, feature, subfeature):
        self._test_api_get_job_result_saved_output(provider, feature, subfeature)

    @pytest.mark.skipif(
        os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
        reason="Skip in opensource package cicd workflow",
    )
    def test_launch_job_invalid_parameters(self, provider, feature, subfeature):
        self._test_launch_job_invalid_parameters(provider, feature, subfeature)

    @pytest.mark.skipif(
        os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
        reason="Skip in opensource package cicd workflow",
    )
    def test_get_job_result_does_not_exist(self, provider, feature, subfeature):
        self._test_get_job_result_does_not_exist(provider, feature, subfeature)


@pytest.mark.skipif(os.environ.get("TEST_SCOPE") != "CICD", reason="Run On CICD")
@pytest.mark.parametrize(
    ("providers", "feature", "subfeature"),
    global_features(filter=only_async_without_phase)["grouped_providers"],
)
class TestFeatureAsyncSubfeatures(CommonAsyncTests):
    def test_async_subfeature(self, providers, feature, subfeature):
        """Test API Call"""
        failures = []
        # List of providers with wokring / not working
        for provider in providers:
            self._test_api_get_job_result_saved_output(provider, feature, subfeature)
            try:
                self._test_async_job(provider, feature, subfeature)
                self._test_get_job_result_does_not_exist(provider, feature, subfeature)
                self._test_launch_job_invalid_parameters(provider, feature, subfeature)
            except Exception as exc:
                print(traceback.format_exc())
                failures.append(exc)

        # Only fail if all providers failes in a certain feature/subfeature
        print(failures)
        assert len(providers) != len(failures)

    def test_async_subfeature_fake(self, providers, feature, subfeature):
        """Fake call"""
        for provider in providers:
            self._test_api_get_job_result_saved_output(provider, feature, subfeature)
