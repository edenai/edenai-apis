from typing import Callable, Any
import pytest

from edenai_apis.interface import list_features, list_providers

only_async = lambda p, f, s, ph: '_async' not in s
without_phase = lambda p, f, s, ph: ph
without_async = lambda p, f, s, ph: '_async' in s
without_async_and_phase = lambda p, f, s, ph: '_async' in s or ph

def global_features(filter: Callable[[Any], bool] = None, return_phase: bool = False):
    """Generate a list of parameters for tests classes.
    Args:
        filter(lambda): filter object
    Returns:
         list [] providers   : [([provider1, provider2], feature, subfeature)]
    """
    method_list = list_features()
    detailed_providers_list = []
    params_dict = {}

    for provider, feature, subfeature, *phase in method_list:
        if filter and filter(provider, feature, subfeature, phase):
            continue
        params_list = [provider, feature, subfeature]

        if return_phase:
            params_list.append(phase[0] if phase else "")

        detailed_params = pytest.param(
            *params_list,
            marks=[
                getattr(pytest.mark, provider),
                getattr(pytest.mark, feature),
                getattr(pytest.mark, subfeature),
            ],
        )
        detailed_providers_list.append(detailed_params)
    
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

def global_providers(filter: Callable[[str], bool] = None):
    """Generate a list of parameters for tests classes.
    Args:
        filter(lambda): filter object
    Returns:
         list [] providers   : [provider1, provider2]
    """
    method_list = list_providers()
    detailed_providers_list = []

    for provider in method_list:
        if filter and filter(provider):
            continue
        detailed_params = pytest.param(
            provider,
            marks=[
                getattr(pytest.mark, provider),
            ]
        )
        detailed_providers_list.append(detailed_params)
    return detailed_providers_list


def pytest_addoption(parser):
    parser.addoption("--provider", action="store", dest="output", help="feature")
    parser.addoption("--feature", action="store", help="feature")
    parser.addoption("--subfeature", action="store", help="subfeature")
    parser.addoption("--phase", action="store", help="phase", default="")


def pytest_configure(config):
    method_list = list_features()
    # params_list = []
    pytest.PROVIDER = "google"
    pytest.FEATURE = "translation"
    pytest.SUBFEATURE = "automatic_translation"
    for provider_i, feature_i, subfeature_i, *phase_i in method_list:
        config.addinivalue_line(
            "markers", provider_i + ": mark test to run only on named provider"
        )
        config.addinivalue_line(
            "markers", feature_i + ": mark test to run only on named feature"
        )
        config.addinivalue_line(
            "markers", subfeature_i + ": mark test to run only on named subfeature"
        )
        if phase_i:
            config.addinivalue_line(
                "markers", phase_i[0] + ": mark test to run only on named phase"
            )


@pytest.fixture
def provider(request):
    return request.config.getoption("--provider")


@pytest.fixture
def feature(request):
    return request.config.getoption("--feature")


@pytest.fixture
def subfeature(request):
    return request.config.getoption("--subfeature")


@pytest.fixture
def phase(request):
    return request.config.getoption("--phase")
