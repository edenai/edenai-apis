from edenai_apis.utils.exception import ProviderException, LanguageException


def test_provider_exception():
    try:
        raise ProviderException(message="Error", code=400)
    except ProviderException as exc:
        assert exc.code == 400
        assert str(exc) == "Error"


def test_language_exception():
    try:
        raise LanguageException(message="Error", code=400)
    except LanguageException as exc:
        assert exc.code == 400
        assert str(exc) == "Error"
