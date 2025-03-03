import pytest

from edenai_apis.features.text.topic_extraction.topic_extraction_dataclass import (
    ExtractedTopic,
)

FEATURE = "text"
SUBFEATURE = "topic_extraction"


class TestTopicExtractionDataClass:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("category", "expected_category"),
        [
            pytest.param(
                "category",
                "Category",
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                "Category",
                "Category",
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                "cateGory",
                "Category",
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
        ],
    )
    def test_valid_category(self, category, expected_category):
        klass = ExtractedTopic(category=category, importance=1)

        assert klass.category == expected_category

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("importance", "expected_importance"),
        [
            pytest.param(
                0,
                0.00,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                1,
                1.00,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                0.578,
                0.58,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
        ],
    )
    def test_valid_importance(self, importance, expected_importance):
        klass = ExtractedTopic(category="category", importance=importance)

        assert klass.importance == expected_importance
