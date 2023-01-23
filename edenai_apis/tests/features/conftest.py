import pytest

def pytest_addoption(parser):
    parser.addoption("--provider", action="store", dest="output", help="feature")
    parser.addoption("--feature", action="store", help="feature")
    parser.addoption("--subfeature", action="store", help="subfeature")
    parser.addoption("--phase", action="store", help="phase", default="")


def pytest_configure():
    pytest.job_id = None

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

# def pytest_configure():
#     pytest.job_id = None
