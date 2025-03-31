import pytest

from edenai_apis.features.text.keyword_extraction import InfosKeywordExtractionDataClass

FEATURE = "text"
SUBFEATURE = "keyword_extraction"


class TestInfosKeywordExtraction:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("importance", "expected_importance"),
        [
            pytest.param(
                0.578,
                0.58,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                None,
                None,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
        ],
    )
    def test_validator_importance(self, importance, expected_importance):
        klass = InfosKeywordExtractionDataClass(keyword="word", importance=importance)

        assert (
            klass.importance == expected_importance
        ), "importance must be rounded to the hundredth"
