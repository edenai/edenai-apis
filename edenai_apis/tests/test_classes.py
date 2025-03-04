"""
Test if providers classes are well formed :
- Inherit from ProviderInterface
- implement a well formatted info.json containing subfeatures versions
- implement all features defined in info.json
"""

from typing import List

import pytest
from _pytest.mark.structures import ParameterSet

from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.interface import list_features
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


def load_class_with_subfeature() -> List[ParameterSet]:
    """Get all Providers Api with feature and subfeature as marker
    returns [
               cls,
               [
                   mark.provider,
                   mark.feature_a, mark.subfeature_1, mark.subfeature_2 ...
                   mark.feature_b, mark.subfeature_1, mark.subfeature_2 ...
               ],
           ]
    """
    api_classes = load_provider(ProviderDataEnum.CLASS)
    method_dict = list_features(as_dict=True)

    api_class_params = []

    for cls in api_classes:
        # Check if provider can be loaded correctly
        if method_dict.get(cls.provider_name):
            # Define marker
            marks = [
                getattr(pytest.mark, cls.provider_name),
            ]
            # List all features for the given provider
            feature_dict = method_dict[cls.provider_name]
            for feature in feature_dict:
                marks.append(getattr(pytest.mark, feature))
                # List all subfeatures from that feature
                for subfeature in feature_dict[feature]:
                    marks.append(getattr(pytest.mark, subfeature))
            api_class_params.append(pytest.param(cls, marks=marks))
    return api_class_params


@pytest.mark.parametrize("cls", load_class_with_subfeature())
class TestApiClass:
    @pytest.mark.unit
    def test_issubclass(self, cls: ProviderInterface):
        assert issubclass(
            cls, ProviderInterface
        ), f"Please inherit {cls} from ProviderInterface"

    @pytest.mark.integration
    def test_info_file_exists(self, cls: ProviderInterface):
        provider = cls.provider_name
        info = load_provider(ProviderDataEnum.INFO_FILE, provider)
        assert info, "info file does not exist"

    @pytest.mark.integration
    def test_version_exists(self, cls: ProviderInterface):
        provider = cls.provider_name
        info = load_provider(ProviderDataEnum.INFO_FILE, provider)
        # exclude _metadata as it's a special field and doesn't represent a feature
        for feature in [f for f in info if f != "_metadata"]:
            for subfeature in info[feature]:
                assert isinstance(
                    info[feature][subfeature], dict
                ), f"`{info[feature][subfeature]}` should be a dict"
                if not info[feature][subfeature].get("version"):
                    for phase in info[feature][subfeature]:
                        assert (
                            "version" in info[feature][subfeature][phase]
                        ), "missing 'version' property"
                else:
                    assert (
                        "version" in info[feature][subfeature]
                    ), "missing 'version' property"

    @pytest.mark.integration
    def test_implemented_features_documented(self, cls: ProviderInterface):
        """Test if all implemented features are documented in the provider's info.json file"""
        # Setup
        provider = cls.provider_name
        info = load_provider(ProviderDataEnum.INFO_FILE, provider)
        implemented_features = list_features(provider_name=provider)

        # Action
        for _, feature, subfeature, *phase in implemented_features:
            if phase:
                feature_info = (
                    info.get(feature, {}).get(subfeature, {}).get(phase[0], {})
                )
            else:
                feature_info = info.get(feature, {}).get(subfeature, {})
            # Assert if feature_info doesn't exist
            assert (
                feature_info
            ), f"Please add {(feature,subfeature,phase[0] if phase else '')} to info.json file of {cls.__name__}"
