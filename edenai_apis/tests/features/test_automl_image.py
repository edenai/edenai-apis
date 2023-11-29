import os
from time import sleep

import pytest

from edenai_apis import Image
from edenai_apis.features.image.automl_classification.create_project_async import (
    AutomlClassificationCreateProjectDataClass,
)
from edenai_apis.interface import list_providers
from edenai_apis.loaders.loaders import load_feature
from edenai_apis.loaders.data_loader import FeatureDataEnum
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.types import AsyncLaunchJobResponseType, AsyncBaseResponseType

automl_image_providers = sorted(
    list_providers(feature="image", subfeature="automl_classification")
)
MAX_TIME = 180
TIME_BETWEEN_CHECK = 10


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Skip in opensource package cicd workflow",
)
@pytest.mark.parametrize(("provider"), automl_image_providers)
@pytest.mark.xdist_group(name="image_automl_classification")
class TestImageAutomlClassification:
    def test_create_project_launch(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="automl_classification",
            phase="create_project_async",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider,
            "image",
            "automl_classification",
            "create_project_async",
            feature_args,
        )
        try:
            create_project_launch = (
                Image.automl_classification__create_project_async__launch_job(provider)
            )
        except AttributeError:
            raise AttributeError("Could not create project async launch phase.")
        create_project_launch_output: AsyncLaunchJobResponseType = (
            create_project_launch(**feature_args).model_dump()
        )

        assert (
            create_project_launch_output.provider_job_id is not None
        ), "Create project async launch job phase failed"
        pytest.job_id = create_project_launch_output.provider_job_id

    def test_create_project_get_job_result(self, provider):
        provider_job_id = pytest.job_id
        try:
            create_project_get = (
                Image.automl_classification__create_project_async__get_job_result(
                    provider
                )
            )
        except AttributeError:
            raise AttributeError("Could not create project async get phase.")
        sleep(5)
        current_time = MAX_TIME
        while current_time > 0:
            create_project_get_output = create_project_get(provider_job_id)
            create_project_get_output_dict = create_project_get_output.model_dump()
            if create_project_get_output_dict["status"] != "pending":
                break
            current_time = current_time - TIME_BETWEEN_CHECK
            sleep(TIME_BETWEEN_CHECK)
        original_response = create_project_get_output.original_response
        standardized_response = create_project_get_output.standardized_response

        assert (
            create_project_get_output_dict.get("status") == "success"
        ), "Create project async get job result phase failed"
        assert isinstance(
            create_project_get_output, AsyncBaseResponseType
        ), f"Expected AsyncBaseResponseType bug got {type(create_project_get_output)}"
        assert isinstance(
            standardized_response, AutomlClassificationCreateProjectDataClass
        ), f"Expected AutomlClassificationCreateProjectDataClass but got {type(standardized_response)}"
        assert original_response is not None
