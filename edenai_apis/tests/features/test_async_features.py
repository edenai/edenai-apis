"""
    Test all async subfeatures for all providers to check if:
    - Job ID is not null
    - Saved output for each provider exists and is well standardized
    - providers APIs work and their outputs are well standardized
"""
from pprint import pprint
import pytest
import os
import traceback
from time import sleep
import importlib
import logging
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.tests.conftest import global_features, only_async
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.types import AsyncBaseResponseType, AsyncLaunchJobResponseType
from edenai_apis.utils.compare import compare_responses


MAX_TIME = 180
TIME_BETWEEN_CHECK = 10
INTERFACE_MODULE = importlib.import_module("edenai_apis.interface_v2")

class CommonAsyncTests:
    def _test_launch_job_id(self, provider, feature, subfeature):
        logging.info(f"Testing launch job id for {provider}, {subfeature}..\n")

        # Step 1 (setup) : prepare parameters 
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature)
        validated_args = validate_all_provider_constraints(provider, feature, subfeature, feature_args)
        try:
            subfeature_suffix = "__launch_job"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_launch_job_function = getattr(feature_class, f"{subfeature}{subfeature_suffix}")(provider)
        except AttributeError:
            raise('Could not import provider launch job method.')

        # Step 2 (action) : Launch the job (action)
        launch_job_response : AsyncLaunchJobResponseType = provider_launch_job_function(**validated_args)

        # Step 3 (assert) : Assert job_id
        assert launch_job_response.provider_job_id is not None, "provider job id should not be null."

        pytest.job_id = launch_job_response.provider_job_id
    
    def _test_api_get_job_result_real_output(self, provider, feature, subfeature):
        logging.info(f"Testing get job result with real output for {provider}, {subfeature}..\n")
        # skip in opensource package cicd workflow
        if os.environ.get("TEST_SCOPE") == 'CICD-OPENSOURCE':
            return

        # Step 1 (setup) : prepare parameters
        provider_job_id = pytest.job_id
        try:
            subfeature_suffix = "__get_job_result"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_get_job_result_method = getattr(feature_class, f"{subfeature}{subfeature_suffix}")(provider)
        except AttributeError:
            raise('Could not import provider get job method.')

        # Step 2 (actions) : call get job result with a valid job id        
        sleep(5)
        current_time = MAX_TIME
        while current_time > 0:
            print(f"wait job result {MAX_TIME- current_time}s")
            provider_api_output = provider_get_job_result_method(provider_job_id)
            provider_api_output_dict = provider_api_output.dict()
            if provider_api_output_dict["status"] != "pending":
                provider_api_output = provider_api_output
                break
            current_time = current_time - TIME_BETWEEN_CHECK
            sleep(TIME_BETWEEN_CHECK)

        pprint(provider_api_output_dict)
        original_response = provider_api_output_dict.get('original_response')
        standardized_response = provider_api_output_dict.get('standardized_response')
        standardized = compare_responses(feature, subfeature, standardized_response)

        # Step 3 (asserts) : check dataclass standardization
        assert isinstance(provider_api_output, AsyncBaseResponseType), f"Expected AsyncBaseResponseType but got {type(provider_api_output)}"
        assert original_response is not None, 'original_response should not be None'
        assert standardized_response is not None, 'standardized_response should not be None'
        assert standardized, 'The output is not standardized' 

    def _test_async_job(self, provider, feature, subfeature):
        self._test_launch_job_id(provider, feature, subfeature)
        self._test_api_get_job_result_real_output(provider, feature, subfeature)
        
    def _test_api_get_job_result_saved_output(self, provider, feature, subfeature):
        logging.info(f"Testing get job result with saved output for {provider}, {subfeature}..\n")

        # Step 1 (Setup) : 
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, feature, subfeature
        )

        # Step 2 (Action) :
        standardized = compare_responses(feature, subfeature, saved_output["standardized_response"])

        # Step 3 (Assert) : 
        assert standardized, 'The output is not standardized'

@pytest.mark.skipif(os.environ.get("TEST_SCOPE") == 'CICD-OPENSOURCE', reason="Don't run on opensource cicd workflow")
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_features(only_async)['ungrouped_providers'],
)
class TestAsyncSubFeatures(CommonAsyncTests):

    def test_async_job(self, provider, feature, subfeature):
        self._test_async_job(provider, feature, subfeature)

    def test_launch_job_invalid_parameters(self, provider, feature, subfeature):
        logging.info(f"Testing launch job with invalid parameters for {provider}, {subfeature}..\n")

        # Step 1 (setup) : prepare invalid parameters
        invalid_feature_args = {'invalid_key': 'invalid_value'}
        try:
            subfeature_suffix = "__launch_job"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_launch_job_function = getattr(feature_class, f"{subfeature}{subfeature_suffix}")(provider)
        except AttributeError:
            raise('Could not import provider launch job method.')
 
        # Step 2 & 3 (action & assert ) : Call the feature with invalid parameters
        with pytest.raises(TypeError):
            provider_launch_job_function(**invalid_feature_args)

    def test_api_get_job_result_saved_output(self, provider, feature, subfeature):
        self._test_api_get_job_result_saved_output(provider, feature, subfeature)
    
    def test_get_job_result_does_not_exist(self, provider, feature, subfeature):
        logging.info(f"Testing get job result with invalid id for {provider}, {subfeature}..\n")
        # Step 1 (setup) : prepare a non-existent job ID
        job_id = '12345678-1234-1234-1234-123456789abc'
        try:
            subfeature_suffix = "__get_job_result"
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_get_job_result_method = getattr(feature_class, f"{subfeature}{subfeature_suffix}")(provider)
        except AttributeError:
            raise('Could not import provider get job method.')

        # Step 2 & 3 (action & assert) : Call the feature with the non-existent job ID
        with pytest.raises(Exception):
            provider_get_job_result_method(provider, feature, subfeature, job_id)

@pytest.mark.skipif(os.environ.get("TEST_SCOPE") != "CICD", reason="Run On CICD")
@pytest.mark.parametrize(
    ("providers", "feature", "subfeature"),
    global_features(filter=only_async)['grouped_providers'],
)
class TestFeatureAsyncSubfeatures(CommonAsyncTests):

    def test_async_subfeature_output(self, providers, feature, subfeature):
        """Test API Call"""
        failures = []
        # List of providers with wokring / not working
        for provider in providers:
            self._test_api_get_job_result_saved_output(provider, feature, subfeature)
            try:
                self._test_async_job(provider, feature, subfeature)
            except Exception as exc:
                print(traceback.format_exc())
                failures.append(exc)

        # Only fail if all providers failes in a certain feature/subfeature
        print(failures)
        assert len(providers) != len(failures)

    def test_sync_subfeature_output(self, providers, feature, subfeature):
        """Fake call"""
        for provider in providers:
            self._test_api_get_job_result_saved_output(provider, feature, subfeature)
