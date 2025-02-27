import pytest

from edenai_apis.utils.exception import LanguageException, ProviderException


@pytest.mark.unit
def test_provider_exception():
    try:
        raise ProviderException(message="Error", code=400)
    except ProviderException as exc:
        assert exc.code == 400
        assert str(exc) == "Error"


@pytest.mark.unit
def test_language_exception():
    try:
        raise LanguageException(message="Error", code=400)
    except LanguageException as exc:
        assert exc.code == 400
        assert str(exc) == "Error"
