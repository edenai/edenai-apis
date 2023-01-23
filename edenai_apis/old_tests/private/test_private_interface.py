"""
    Test interface functions :
    - compute_output
    - list_features
    - list_providers
    - check_provider_constraints
    This file should be equivalent to test_interface.py except
    that functionss should be loaded from edenai_apis.private_interface
"""
from typing import Dict
from pytest_mock import MockerFixture
from edenai_apis.private_interface import (
    check_provider_constraints,
    compute_output,
    list_features,
    list_providers,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType



VALID_PROVIDER = "amazon"
VALID_FEATURE = "audio"
VALID_SUBFEATURE = "text_to_speech"


class TestComputeOutput:
    exception_name = "EXCEPTION"

    def test_output(self, mocker: MockerFixture):
        def fake_load_provider(*args, **kwargs):
            def faker(**kwargs):
                return ResponseType[Dict](original_response={}, standardized_response={})
            return faker
        def fake_load_provider_info_file(*args, **kwargs):
            return {}

        mocker.patch(
            "edenai_apis.interface.load_provider", side_effect=fake_load_provider
        )
        mocker.patch(
            "edenai_apis.utils.constraints.load_provider", side_effect=fake_load_provider_info_file
        )

        final_result = compute_output("p", "f", "s", {})
        assert final_result
        assert final_result["status"] == "success"

        final_result_async = compute_output("p", "f", "s_async", {})
        assert final_result_async
        assert final_result["status"] == "success"

        final_result = compute_output("p", "f", "s", {})
        assert final_result
        assert final_result["status"] == "success"

        final_result_async = compute_output("p", "f", "s_async", {})
        assert final_result_async
        assert final_result["status"] == "success"


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
