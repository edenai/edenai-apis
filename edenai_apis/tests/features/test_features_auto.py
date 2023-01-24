"""
    Test all synchronous subfeatures (except image search and face regonition) for all providers to check if:
    - Valid input data
    - Saved output for each provider exists and is well standardized
    - providers APIs work and their outputs are well standardized
"""
import pytest
import importlib
import datetime
import json
from edenai_apis.interface import list_features
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.utils.compare import compare_responses
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


INTERFACE_MODULE = importlib.import_module("edenai_apis.interface_v2")
def global_features():
    """Generate a list of parameters for tests classes.
    Returns:
         list [] providers   : [([provider1, provider2], feature, subfeature)]
    """
    method_list = list_features()
    detailed_providers_list = []

    for provider, feature, subfeature, *phase in method_list:
        if '_async' in subfeature or phase:
            continue
        detailed_params = pytest.param(
            provider,
            feature,
            subfeature,
            marks=[
                getattr(pytest.mark, provider),
                getattr(pytest.mark, feature),
                getattr(pytest.mark, subfeature)],
        )
        detailed_providers_list.append(detailed_params)
    return detailed_providers_list

@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_features(),
)
class TestSubfeatures:
    def test_feature_with_invalid_file(self, provider, feature, subfeature):
        # Setup
        invalid_file = "fakefile.txt"
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature)
        if not feature_args.get('file'):
            pytest.skip("unsupported configuration")
        feature_args['file'] = invalid_file
        
        # Action
        with pytest.raises(ProviderException, AttributeError) as exc:
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_method = getattr(feature_class, f"{subfeature}")(provider)
            provider_method(**feature_args)
            assert exc is not None, 'ProviderException or AttributeError expected.'

    
    def test_feature_api_call(self, provider, feature, subfeature):
        # Setup
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature)
        validated_args = validate_all_provider_constraints(provider, feature, subfeature, feature_args)
        try:
            feature_class = getattr(INTERFACE_MODULE, feature.capitalize())
            provider_method = getattr(feature_class, f"{subfeature}")(provider)
        except AttributeError:
            raise('Could not import provider method.')
        
        # Actions
        provider_api_output = provider_method(**validated_args)
        provider_api_dict = provider_api_output.dict()
        original_response = provider_api_dict.get('original_response')
        standardized_response = provider_api_dict.get('standardized_response')
        standardized = compare_responses(feature, subfeature, standardized_response)
        
        # Step 3 (asserts) : check dataclass standardization
        assert isinstance(provider_api_output, ResponseType), f"Expected ResponseType but got {type(provider_api_output)}"
        assert original_response is not None, 'original_response should not be None'
        assert standardized_response is not None, 'standardized_response should not be None'
        assert standardized, 'The output is not standardized' 
        
    def test_feature_saved_output(self, provider, feature, subfeature):
        # Step 1 (Setup) : 
        saved_output = load_provider(
            ProviderDataEnum.OUTPUT, provider, feature, subfeature
        )
        
        # Step 2 (Action) :
        standardized = compare_responses(feature, subfeature, saved_output["standardized_response"])
        
        # Step 3 (Assert) : 
        assert standardized, 'The output is not standardized'