import pytest
from pytest_mock import MockerFixture

from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider


class TestLoadFeature:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("data_feature"), [FeatureDataEnum.DATA_CLASS, FeatureDataEnum.SAMPLES_ARGS]
    )
    def test_valid_load_feature_feature(
        self, mocker: MockerFixture, data_feature: FeatureDataEnum
    ):
        mocker_loader = mocker.patch(
            f"edenai_apis.loaders.data_loader.{data_feature.value}"
        )
        load_feature(
            data_feature=data_feature,
        )

        mocker_loader.assert_called_once()


class TestLoadProvider:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("data_provider"),
        [
            ProviderDataEnum.CLASS,
            ProviderDataEnum.INFO_FILE,
            ProviderDataEnum.KEY,
            ProviderDataEnum.OUTPUT,
            ProviderDataEnum.PROVIDER_INFO,
            ProviderDataEnum.SUBFEATURE,
        ],
    )
    def test_valid_load_provider(
        self, mocker: MockerFixture, data_provider: ProviderDataEnum
    ):
        mocker_loader = mocker.patch(
            f"edenai_apis.loaders.data_loader.{data_provider.value}"
        )
        load_provider(
            data_provider=data_provider,
        )

        mocker_loader.assert_called_once()
