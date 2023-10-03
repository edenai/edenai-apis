import pytest

from edenai_apis.features.text.moderation.moderation_dataclass import (
    ModerationDataClass,
)

FEATURE = "text"
SUBFEATURE = "moderation"


class TestTextModeration:
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
    def test_valid_value_check_min_mac_nsfw(self, nsfw_likelihood, nsfw_likelihood_score):
        try:
            ModerationDataClass(nsfw_likelihood=nsfw_likelihood, items=[], nsfw_likelihood_score=nsfw_likelihood_score)
        except ValueError:
            pytest.fail(f"{nsfw_likelihood} value doesn't raises a ValueError")
