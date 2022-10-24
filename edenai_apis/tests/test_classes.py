"""
    Test if providers classes are well formed :
    - Inherit from ProviderApi
    - implement a well formatted info.json containing subfeatures versions
    - implement all features defined in info.json
"""
from typing import List
import pytest
from _pytest.mark.structures import ParameterSet

from edenai_apis.features.base_provider.provider_api import ProviderApi
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
                pytest.mark.cls,
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
    def test_info_provider_api(self, cls: ProviderApi):
        """Test if ProviderApi has basic infos like version for all feature implemented"""

        assert issubclass(
            cls, ProviderApi
        ), f"Please inherit {cls} from ProviderApi"

        provider = cls.provider_name

        # Check if there is an info file
        info = load_provider(ProviderDataEnum.INFO_FILE, provider)
        assert info

        # list all features, subfeatures, *phase implemented for the provider's class
        implemented_feature_subfeature = list_features(provider_name=provider)

        # list of all (features, subfeatures, *phase) defined in the provider's info.json file
        info_feature_subfeatures = []

        # Check if all features in info.json have a version
        for feature in info:
            for subfeature in info[feature]:
                # `version` is at the deepest level, if it is not,
                # it means either there is a phase or it was not written
                if not info[feature][subfeature].get("version"):
                    for phase in info[feature][subfeature]:
                        info_feature_subfeatures.append((feature, subfeature, phase))
                        assert (
                            "version" in info[feature][subfeature][phase]
                        ), "missing 'version' property"
                        assert (
                            provider,
                            feature,
                            subfeature,
                            phase,
                        ) in implemented_feature_subfeature,\
                            f"{(provider, feature, subfeature, phase)} in info but not implemented"
                else:
                    info_feature_subfeatures.append((feature, subfeature))
                    assert "version" in info[feature][subfeature]
                    assert (
                        provider,
                        feature,
                        subfeature,
                    ) in implemented_feature_subfeature, (
                        f"{(provider, feature, subfeature)} in info but not implemented"
                    )

        # TO DO: Add language constraint tests if textual feature

        # Check if all implemented features are documented in info.json
        for _, feature, subfeature, *phase in implemented_feature_subfeature:
            if not phase:
                assert (
                    feature,
                    subfeature,
                ) in info_feature_subfeatures, f"Please add {(feature,subfeature)}\
                    to info.json file of {cls.__name__}"
            else:
                assert (feature, subfeature, phase[0],) in info_feature_subfeatures, (
                    f"Please add {(feature,subfeature,phase[0])}"
                    + f"to info.json file of {cls.__name__}"
                )
