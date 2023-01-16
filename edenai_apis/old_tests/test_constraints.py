import os

import pytest
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.settings import tests_path
from edenai_apis.old_tests.test_features_auto import global_method_list_as_params
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.exception import ProviderException


@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_method_list_as_params()["ungrouped_providers"],
)
def test_file_type_constraints_on_all_providers_wrong_file_type(provider, feature, subfeature):
    constraints = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        provider_name=provider,
        feature=feature,
        subfeature=subfeature,
    ).get("constraints")

    # test only when there is file types constraints
    if constraints is not None and "file_types" in constraints:
        file_with_wrong_file_type_path = os.path.join(
            tests_path, "data", "wrong_file_type_constraints.jar"
        )
        with open(file_with_wrong_file_type_path, "bw") as f:
            # arguments that should fail all constraints
            args = {
                "file": f,  # using a filetype we know will fail all
            }
        with pytest.raises(ProviderException):
            validate_all_provider_constraints(provider, feature, subfeature, args)

@pytest.mark.parametrize(
    ("provider", "feature", "subfeature"),
    global_method_list_as_params()["ungrouped_providers"],
)
def test_language_constraints_on_all_providers_wrong_language(provider, feature, subfeature):
    constraints = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        provider_name=provider,
        feature=feature,
        subfeature=subfeature,
    ).get("constraints")

    # test only when there is language constraints
    if constraints is not None and ("languages" in constraints):
        # arguments that should fail all constraints
        args = {
            "language": "wrong",
            "source_language": "wrong",
            "target_language": "wrong",
        }
        with pytest.raises(ProviderException):
            validate_all_provider_constraints(provider, feature, subfeature, args)
