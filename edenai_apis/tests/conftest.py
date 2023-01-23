import pytest

from edenai_apis.interface import list_features


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
