import pytest

from edenai_apis.features.text.moderation.moderation_dataclass import (
    ModerationDataClass,
)

FEATURE = "text"
SUBFEATURE = "moderation"


class TestTextModeration:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("nsfw_likelihood"),
        [
            pytest.param(
                0,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                5,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                2,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("nsfw_likelihood_score"),
        [
            pytest.param(
                0,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                0.2,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
            pytest.param(
                0.5,
                marks=[getattr(pytest.mark, FEATURE), getattr(pytest.mark, SUBFEATURE)],
            ),
        ],
    )
    def test_valid_value_check_min_mac_nsfw(
        self, nsfw_likelihood, nsfw_likelihood_score
    ):
        try:
            ModerationDataClass(
                nsfw_likelihood=nsfw_likelihood,
                items=[],
                nsfw_likelihood_score=nsfw_likelihood_score,
            )
        except ValueError:
            pytest.fail(f"{nsfw_likelihood} value doesn't raises a ValueError")

    @pytest.mark.unit
    def test_text_moderation_items(self):

        ModerationDataClass(
            nsfw_likelihood=0,
            items=[
                {
                    "label": "hi",
                    "category": "Toxic",
                    "likelihood": 1,
                    "subcategory": "Toxic",
                    "likelihood_score": 0.5,
                }
            ],
            nsfw_likelihood_score=0.0,
        )
