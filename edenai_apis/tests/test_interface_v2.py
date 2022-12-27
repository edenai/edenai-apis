import importlib
import random

import pytest
from edenai_apis.interface import list_features
from edenai_apis.interface_v2 import abstract, return_provider_method
from edenai_apis.loaders.data_loader import FeatureDataEnum, load_subfeature
from edenai_apis.loaders.loaders import load_feature

# Unify tuples so they all have a length of 4
_correct_list_feature = lambda: map(
    lambda flist: [*flist, ""] if len(flist) == 3 else flist, list_features()
)


@pytest.mark.parametrize(
    ("provider", "feature", "subfeature", "phase"), _correct_list_feature()
)
def test_return_provider_method_returns_right_method(
    provider, feature, subfeature, phase
):
    """Take a random method from a random feature interface"""

    subfeature_methods = [subfeature]
    if '_async' in subfeature:
        subfeature_methods.append(subfeature)
        subfeature_methods[0] += '__launch_job'
        subfeature_methods[1] += '__get_job_result'
    phase_suffix = f"__{phase}" if phase else ""

    for method_name in subfeature_methods:

        feature_interface_module = importlib.import_module(
            f"edenai_apis.features.{feature}.{feature}_interface"
        )
        provider_interface_class = getattr(
            feature_interface_module, f"{feature.capitalize()}Interface"
        )

        assert hasattr(provider_interface_class, f"{feature}__{method_name}{phase_suffix}")

        tested_interface_method = getattr(
            provider_interface_class, f"{feature}__{method_name}{phase_suffix}"
        )

        right_method = load_subfeature(provider, feature, method_name, phase)

        wrapped = return_provider_method(tested_interface_method)
        assert callable(wrapped)

        test_method = wrapped(provider)
        assert callable(test_method)
        assert test_method.__name__ == right_method.__name__

        # assert test_method has same class as right_method
        assert isinstance(test_method.__self__, type(right_method.__self__))


@pytest.mark.parametrize(
    ("provider", "feature", "subfeature", "phase"), _correct_list_feature()
)
def test_abstract_returns_right_method(provider, feature, subfeature, phase):
    """Test the abstract method that create Interfaces"""
    subfeature_methods = [subfeature]
    if '_async' in subfeature:
        subfeature_methods.append(subfeature)
        subfeature_methods[0] += '__launch_job'
        subfeature_methods[1] += '__get_job_result'
    phase_suffix = f"__{phase}" if phase else ""

    for method_name in subfeature_methods:
        provider_interface_module = importlib.import_module(
            f"edenai_apis.features.{feature}.{feature}_interface"
        )
        provider_interface_class = getattr(
            provider_interface_module, f"{feature.capitalize()}Interface"
        )
        Interface = abstract(provider_interface_class, method_prefix=f"{feature}__")
        assert f"{method_name}{phase_suffix}" in dir(Interface)


def test_random_interface_call():
    """
    take a random provider/feature/subfeature and use InterfaceV2 to make a call
    """

    provider, feature, subfeature, *phase = random.choice(list_features())
    if "async" in subfeature:
        subfeature += "__launch_job"
    phase_suffix = f"__{phase}" if phase else ""

    interface_module = importlib.import_module("edenai_apis.interface_v2")
    klass = getattr(interface_module, feature.capitalize())
    feature_method = getattr(klass, f"{subfeature}{phase_suffix}")(provider)
    feature_args = load_feature(
        FeatureDataEnum.SAMPLES_ARGS,
        feature=feature,
        subfeature=subfeature,
        phase=phase[0] if phase else "",
    )

    response = feature_method(**feature_args)

    assert response is not None
