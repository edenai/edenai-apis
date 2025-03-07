import os
from typing import Optional

import pytest
from pydantic import BaseModel

from edenai_apis.interface import list_features, list_providers
from edenai_apis.loaders.data_loader import (
    load_class,
    load_dataclass,
    load_info_file,
    load_key,
    load_output,
    load_provider_subfeature_info,
    load_samples,
    load_subfeature,
)
from edenai_apis.tests.conftest import (
    global_features,
    global_providers,
    only_async,
    without_async,
)


def _get_feature_subfeature_phase():
    load_feature_list = list_features()
    list_without_provider = list(
        set([(f, s, ph[0] if ph else "") for (p, f, s, *ph) in load_feature_list])
    )
    detailed_providers_list = []
    for feature, subfeature, *phase in list_without_provider:
        detailed_params = pytest.param(
            feature,
            subfeature,
            phase[0] if phase else "",
            marks=[getattr(pytest.mark, feature), getattr(pytest.mark, subfeature)],
        )
        detailed_providers_list.append(detailed_params)
    return sorted(detailed_providers_list)


class TestLoadKey:
    @pytest.mark.e2e
    @pytest.mark.parametrize(("provider"), sorted(global_providers()))
    def test_load_key_of_valid_provider(self, provider: str):
        if provider == "faker":
            pytest.skip("unsupported provider")
        data = load_key(provider, False)
        assert isinstance(data, dict) or isinstance(
            data, list
        ), f"No settings.json file found for {provider}"


class TestLoadClass:
    @pytest.mark.unit
    @pytest.mark.parametrize(("provider"), sorted(global_providers()))
    def test_load_class_with_all_provider(self, provider: Optional[str]):
        klass = load_class(provider)

        assert klass is not None

    @pytest.mark.unit
    def test_load_class_with_bad_provider(self):
        with pytest.raises(
            ValueError, match="No ProviderInterface class implemented for provider:"
        ):
            load_class("NotAProvider")

    @pytest.mark.unit
    def test_load_class_with_none_provider(self):
        klass = load_class()

        len_klass = len(klass)
        nb_providers = len(list_providers())
        assert len_klass == nb_providers


class TestLoadDataclass:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("feature", "subfeature", "phase"), _get_feature_subfeature_phase()
    )
    def test_load_dataclass(self, feature, subfeature, phase):
        if phase == "create_project":
            pytest.skip(
                "image-search-create_project because this method don't need to return a dataclass"
            )

        dataclass = load_dataclass(feature, subfeature, phase)

        assert issubclass(dataclass, BaseModel)


class TestLoadInfoFile:
    @pytest.mark.unit
    @pytest.mark.parametrize(("provider"), sorted(global_providers()))
    def test_load_info_file_with_one_provider(self, provider):
        info = load_info_file(provider)

        assert isinstance(info, dict)


class TestLoadProviderSubfeatureInfo:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature", "phase"),
        global_features(return_phase=True)["ungrouped_providers"],
    )
    def test_load_info_subfeature_provider(self, provider, feature, subfeature, phase):
        info = load_provider_subfeature_info(provider, feature, subfeature, phase)

        assert isinstance(info, dict)
        assert info.get("version")


class TestLoadOutput:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature", "phase"),
        global_features(return_phase=True)["ungrouped_providers"],
    )
    def test_load_output_valid_paramters(self, provider, feature, subfeature, phase):
        # skip create and delete method
        if "create" in phase or "delete" in phase or "upload" in phase:
            pytest.skip("create, delete and upload phase don't have a output.json")

        output = load_output(provider, feature, subfeature, phase)

        assert isinstance(output, dict), "output should be a dict"
        try:
            if feature == "llm":
                output["choices"]
            else:
                output["original_response"]
                output["standardized_response"]
        except KeyError:
            pytest.fail("Original_response and standradized_response not found")


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Don't run on opensource cicd workflow",
)
class TestLoadSubfeature:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature", "phase"),
        global_features(without_async, return_phase=True)["ungrouped_providers"],
    )
    def test_load_subfeature_sync_subfeature(
        self, provider, feature, subfeature, phase
    ):
        method_subfeature = load_subfeature(provider, feature, subfeature, phase)

        expected_name = f"{feature}__{subfeature}"
        if phase:
            expected_name += f"__{phase}"

        assert callable(method_subfeature)
        assert method_subfeature.__name__ == expected_name

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature", "phase"),
        global_features(only_async, return_phase=True)["ungrouped_providers"],
    )
    def test_load_subfeature_async_subfeature_get_job_result(
        self, provider, feature, subfeature, phase
    ):
        method_subfeature = load_subfeature(
            provider, feature, subfeature, phase, "__get_job_result"
        )

        expected_name = (
            f"{feature}__{subfeature}__get_job_result"
            if not phase
            else f"{feature}__{subfeature}__{phase}__get_job_result"
        )

        assert callable(method_subfeature)
        assert method_subfeature.__name__ == expected_name

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature", "phase"),
        global_features(only_async, return_phase=True)["ungrouped_providers"],
    )
    def test_load_subfeature_async_subfeature_launch_job(
        self, provider, feature, subfeature, phase
    ):
        method_subfeature = load_subfeature(
            provider, feature, subfeature, phase, "__launch_job"
        )

        expected_name = (
            f"{feature}__{subfeature}__launch_job"
            if not phase
            else f"{feature}__{subfeature}__{phase}__launch_job"
        )

        if method_subfeature.__name__ == "func_wrapper":
            assert method_subfeature.__closure__ is not None
            assert len(method_subfeature.__closure__) == 1
            closure = method_subfeature.__closure__[0]
            wrapped_func = closure.cell_contents
            assert wrapped_func.__name__ == expected_name
        else:
            assert callable(method_subfeature)
            assert method_subfeature.__name__ == expected_name


class TestLoadSamples:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("feature", "subfeature", "phase"), _get_feature_subfeature_phase()
    )
    def test_load_sample_valid_parameters(self, feature, subfeature, phase):
        if feature == "image" and (
            subfeature == "search"
            or subfeature == "automl_classification"
            or subfeature == "generation_fine_tuning"
        ):
            pytest.skip(
                "image-search and image-automl_classification need dynamic argument as each provider should have a different project_id"
            )

        args = load_samples(feature, subfeature, phase)

        assert isinstance(args, dict), "Arguments should be a dictionnary"
