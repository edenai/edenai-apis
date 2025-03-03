import pytest

from edenai_apis.features.translation.language_detection import (
    InfosLanguageDetectionDataClass,
)


class TestInfosLanguageDetectionDataClass:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("confidence", "expected_confidence"),
        [
            pytest.param(
                0.0000,
                0.00,
                marks=[pytest.mark.translation, pytest.mark.language_detection],
            ),
            pytest.param(
                1, 1.00, marks=[pytest.mark.translation, pytest.mark.language_detection]
            ),
            pytest.param(
                0.578,
                0.58,
                marks=[pytest.mark.translation, pytest.mark.language_detection],
            ),
        ],
    )
    def test_rounding_confidence_socre(self, confidence, expected_confidence):
        instance = InfosLanguageDetectionDataClass(
            language="valid_language", display_name="valid name", confidence=confidence
        )

        assert instance.confidence == expected_confidence
