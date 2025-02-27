import importlib
import os

import pytest

from edenai_apis.interface import list_features
from edenai_apis.interface_v2 import abstract, return_provider_method
from edenai_apis.loaders.data_loader import load_subfeature

# Unify tuples so they all have a length of 4
_correct_list_feature = lambda: map(
    lambda flist: [*flist, ""] if len(flist) == 3 else flist, list_features()
)


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature", "phase"), _correct_list_feature()
)
def test_return_provider_method_returns_right_method(
    provider, feature, subfeature, phase
):
    """Take a random method from a random feature interface"""

    subfeature_methods = [subfeature]
    if "_async" in subfeature:
        subfeature_methods.append(subfeature)
        subfeature_methods[0] += "__launch_job"
        subfeature_methods[1] += "__get_job_result"
    if phase:
        if "_async" in phase:
            subfeature_methods.append(subfeature)
            subfeature_methods[0] += f"__{phase}__launch_job"
            subfeature_methods[1] += f"__{phase}__get_job_result"
            phase = None
            phase_suffix = ""
        else:
            phase_suffix = f"__{phase}"
    else:
        phase_suffix = ""

    for method_name in subfeature_methods:
        feature_interface_module = importlib.import_module(
            f"edenai_apis.features.{feature}.{feature}_interface"
        )
        provider_interface_class = getattr(
            feature_interface_module, f"{feature.capitalize()}Interface"
        )

        assert hasattr(
            provider_interface_class, f"{feature}__{method_name}{phase_suffix}"
        )

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


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("provider", "feature", "subfeature", "phase"), _correct_list_feature()
)
def test_abstract_returns_right_method(provider, feature, subfeature, phase):
    """Test the abstract method that create Interfaces"""
    subfeature_methods = [subfeature]
    if "_async" in subfeature:
        subfeature_methods.append(subfeature)
        subfeature_methods[0] += "__launch_job"
        subfeature_methods[1] += "__get_job_result"
    if phase:
        if "_async" in phase:
            subfeature_methods.append(subfeature)
            subfeature_methods[0] += f"__{phase}__launch_job"
            subfeature_methods[1] += f"__{phase}__get_job_result"
            phase_suffix = ""
        else:
            phase_suffix = f"__{phase}"
    else:
        phase_suffix = ""

    for method_name in subfeature_methods:
        provider_interface_module = importlib.import_module(
            f"edenai_apis.features.{feature}.{feature}_interface"
        )
        provider_interface_class = getattr(
            provider_interface_module, f"{feature.capitalize()}Interface"
        )
        Interface = abstract(provider_interface_class, method_prefix=f"{feature}__")
        assert f"{method_name}{phase_suffix}" in dir(Interface)
