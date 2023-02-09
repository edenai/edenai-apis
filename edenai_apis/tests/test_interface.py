"""
    Test interface functions :
    - compute_output
    - list_features
    - list_providers
    - check_provider_constraints
"""
from typing import Dict
import pytest
from pytest_mock import MockerFixture
from edenai_apis.tests.conftest import global_features
from edenai_apis.interface import (
    check_provider_constraints,
    compute_output,
    list_features,
    list_providers,
)
from edenai_apis.utils.types import ResponseType



VALID_PROVIDER = "amazon"
VALID_FEATURE = "audio"
VALID_SUBFEATURE = "text_to_speech"


@pytest.mark.parametrize(
    ('provider', 'feature', 'subfeature', 'phase'),
    global_features(return_phase=True)["ungrouped_providers"],
)
class TestComputeOutput:
    def test_output_fake(self, mocker: MockerFixture, provider, feature, subfeature, phase):
        if phase == 'create_project':
            pytest.skip("create_project is not supported in fake mode")
        mocker.patch('edenai_apis.interface.validate_all_provider_constraints', return_value={})
        mocker.patch('edenai_apis.interface.assert_equivalent_dict', return_value=True)
        final_result = compute_output(provider, feature, subfeature, {}, fake=True, phase=phase)
        assert final_result['provider'] == provider
        assert final_result["status"] == "success"

    # NOT WORKING ATM
    # def test_output_real(self, mocker: MockerFixture, provider, feature, subfeature, phase):
    #     if phase == 'create_project':
    #         pytest.skip("create_project is not supported in fake mode")
    #     mocker.patch('edenai_apis.interface.validate_all_provider_constraints', return_value={})
    #     mocker.patch('edenai_apis.interface.assert_equivalent_dict', return_value=True)
    #     mocker.patch(f'edenai_apis.interface.{feature.title()}.{subfeature}{"__" if phase else ""}{phase}', return_value={'data': 'data'})
    #     final_result = compute_output(provider, feature, subfeature, {}, fake=False, phase=phase)
    #     assert final_result['provider'] == provider
    #     assert final_result["status"] == "success"
    #     assert final_result['data'] == 'data'


def test_list_features():
    # with a list as return
    method_list = list_features()
    assert len(method_list) != 0
    for method in method_list:
        assert method
        assert len(method) == 3 or len(method) == 4
        if len(method) == 4:
            assert "face_recognition" in method[2] or "search" in method[2]
        for elem in method:
            assert elem

    # with a dict as return
    method_dict = list_features(as_dict=True)
    assert len(method_dict) != 0
    for provider in method_dict:
        assert provider
        for feature in method_dict[provider]:
            assert feature
            for subfeature in method_dict[provider][feature]:
                assert subfeature
                assert method_dict[provider][feature][subfeature]


def test_list_providers():
    # all providers
    providers = list_providers()
    assert isinstance(providers, list)
    all_provider_length = len(providers)

    # By feature
    providers_by_feature = list_providers(VALID_FEATURE)
    assert isinstance(providers_by_feature, list)
    providers_by_feature_length = len(providers_by_feature)
    assert providers_by_feature_length <= all_provider_length

    # By subfeature
    providers_by_subfeature = list_providers(subfeature=VALID_SUBFEATURE)
    assert isinstance(providers_by_subfeature, list)
    providers_by_subfeature_length = len(providers_by_subfeature)
    assert providers_by_subfeature_length <= all_provider_length

    # By feature and subfeature
    providers_by_feat_subfeat = list_providers(VALID_FEATURE, VALID_SUBFEATURE)
    assert isinstance(providers_by_feat_subfeat, list)
    providers_by_feat_subfeat_length = len(providers_by_feat_subfeat)
    assert (
        providers_by_feat_subfeat_length <= providers_by_feature_length
        and providers_by_feat_subfeat_length <= providers_by_subfeature_length
    )


def test_check_provider_constraints():
    assert check_provider_constraints(VALID_PROVIDER, VALID_FEATURE, VALID_SUBFEATURE)[
        0
    ]
