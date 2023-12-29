import os
from time import sleep

import pytest

from edenai_apis import Image
from edenai_apis.features.image import (
    AutomlClassificationCreateProjectDataClass,
    AutomlClassificationUploadDataAsyncDataClass,
    AutomlClassificationTrainAsyncDataClass,
    AutomlClassificationPredictAsyncDataClass,
    AutomlClassificationDeleteProjectDataClass,
)
from edenai_apis.interface import list_providers
from edenai_apis.loaders.loaders import load_feature
from edenai_apis.loaders.data_loader import FeatureDataEnum
from edenai_apis.utils.constraints import validate_all_provider_constraints
from edenai_apis.utils.types import (
    ResponseType,
    AsyncLaunchJobResponseType,
    AsyncResponseType,
)

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
    def test_create_project(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="automl_classification",
            phase="create_project",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider,
            "image",
            "automl_classification",
            "create_project",
            feature_args,
        )
        try:
            create_project_method = Image.automl_classification__create_project(
                provider
            )
        except AttributeError:
            raise AttributeError("Could not import create project phase.")

        create_project_output = create_project_method(**feature_args)
        original_response = create_project_output.original_response
        standardized_response = create_project_output.standardized_response

        assert isinstance(
            create_project_output, ResponseType
        ), f"Expected ResponseType but got {type(create_project_output)}"
        assert original_response is not None, "Original response should not be empty."
        assert isinstance(
            standardized_response, AutomlClassificationCreateProjectDataClass
        ), f"Expected AutomlClassificationCreateProjectDataClass but got {type(standardized_response)}"

        pytest.project_id = standardized_response.project_id

    def test_upload_data_launch(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="automl_classification",
            phase="upload_data_async",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider,
            "image",
            "automl_classification",
            "upload_data_async",
            feature_args,
        )
        try:
            upload_data_method = (
                Image.automl_classification__upload_data_async__launch_job(provider)
            )
        except AttributeError:
            raise AttributeError("Could not import upload image launch phase.")

        feature_args["project_id"] = pytest.project_id

        upload_data_output = upload_data_method(**feature_args)

        assert isinstance(
            upload_data_output, AsyncLaunchJobResponseType
        ), f"Expected AsyncLaunchJobResponseType but got {type(upload_data_output)}"
        assert (
            upload_data_output.provider_job_id is not None
        ), "provider job is should not be null"
        pytest.job_id = upload_data_output.provider_job_id

    def test_upload_data_get_result(self, provider):
        provider_job_id = pytest.job_id

        try:
            upload_data_method = (
                Image.automl_classification__upload_data_async__get_job_result(provider)
            )
        except AttributeError:
            raise AttributeError("Could not import upload data get job result phase.")
        sleep(5)
        current_time = MAX_TIME
        while current_time > 0:
            print(f"wait upload data job result {MAX_TIME- current_time}s")
            upload_data_output = upload_data_method(provider_job_id)
            upload_data_output_dict = upload_data_output.model_dump()
            if upload_data_output_dict["status"] != "pending":
                upload_data_output = upload_data_output
                break
            current_time = current_time - TIME_BETWEEN_CHECK
            sleep(TIME_BETWEEN_CHECK)
        original_response = upload_data_output.original_response
        standardized_response = upload_data_output.standardized_response
        assert isinstance(
            upload_data_output, AsyncResponseType
        ), f"Expected AsyncResponseType but got {type(upload_data_output)}"
        assert original_response is not None, "Original response should not be empty."
        assert isinstance(
            standardized_response, AutomlClassificationUploadDataAsyncDataClass
        ), f"Expected AutomlClassificationUploadDataAsyncClass but got {type(standardized_response)}"

    def test_train_launch(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="automl_classification",
            phase="train_async",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider,
            "image",
            "automl_classification",
            "train_async",
            feature_args,
        )
        try:
            train_method = Image.automl_classification__train_async__launch_job(
                provider
            )
        except AttributeError:
            raise AttributeError("Could not import train launch phase.")

        train_output = train_method(**feature_args)

        assert isinstance(
            train_output, AsyncLaunchJobResponseType
        ), f"Expected AsyncLaunchJobResponseType but got {type(train_output)}"
        assert (
            train_output.provider_job_id is not None
        ), "provider job is should not be null"
        pytest.job_id = train_output.provider_job_id

    def test_train_get_result(self, provider):
        provider_job_id = pytest.job_id

        try:
            train_method = Image.automl_classification__train_async__get_job_result(
                provider
            )
        except AttributeError:
            raise AttributeError("Could not import train get job result phase.")
        sleep(5)
        current_time = MAX_TIME
        while current_time > 0:
            print(f"wait train job result {MAX_TIME- current_time}s")
            train_output = train_method(provider_job_id)
            train_output_dict = train_output.model_dump()
            if train_output_dict["status"] != "pending":
                train_output = train_output
                break
            current_time = current_time - TIME_BETWEEN_CHECK
            sleep(TIME_BETWEEN_CHECK)
        original_response = train_output.original_response
        standardized_response = train_output.standardized_response
        assert isinstance(
            train_output, AsyncResponseType
        ), f"Expected AsyncResponseType but got {type(train_output)}"
        assert original_response is not None, "Original response should not be empty."
        assert isinstance(
            standardized_response, AutomlClassificationTrainAsyncDataClass
        ), f"Expected AutomlClassificationTrainAsyncClass but got {type(standardized_response)}"

    def test_predict_launch(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="automl_classification",
            phase="predict_async",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider,
            "image",
            "automl_classification",
            "predict_async",
            feature_args,
        )
        try:
            predict_method = Image.automl_classification__predict_async__launch_job(
                provider
            )
        except AttributeError:
            raise AttributeError("Could not import predict launch phase.")

        predict_output = predict_method(**feature_args)

        assert isinstance(
            predict_output, AsyncLaunchJobResponseType
        ), f"Expected AsyncLaunchJobResponseType but got {type(predict_output)}"
        assert (
            predict_output.provider_job_id is not None
        ), "provider job is should not be null"
        pytest.job_id = predict_output.provider_job_id

    def test_predict_get_result(self, provider):
        provider_job_id = pytest.job_id

        try:
            predict_method = Image.automl_classification__predict_async__get_job_result(
                provider
            )
        except AttributeError:
            raise AttributeError("Could not import predict get job result phase.")
        sleep(5)
        current_time = MAX_TIME
        while current_time > 0:
            print(f"wait predict job result {MAX_TIME- current_time}s")
            predict_output = predict_method(provider_job_id)
            predict_output_dict = predict_output.model_dump()
            if predict_output_dict["status"] != "pending":
                predict_output = predict_output
                break
            current_time = current_time - TIME_BETWEEN_CHECK
            sleep(TIME_BETWEEN_CHECK)
        original_response = predict_output.original_response
        standardized_response = predict_output.standardized_response
        assert isinstance(
            predict_output, AsyncResponseType
        ), f"Expected AsyncResponseType but got {type(predict_output)}"
        assert original_response is not None, "Original response should not be empty."
        assert isinstance(
            standardized_response, AutomlClassificationPredictAsyncDataClass
        ), f"Expected AutomlClassificationPredictAsyncClass but got {type(standardized_response)}"

    def test_delete_project(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="image",
            subfeature="automl_classification",
            phase="delete_project",
            provider_name=provider,
        )
        feature_args = validate_all_provider_constraints(
            provider,
            "image",
            "automl_classification",
            "delete_project",
            feature_args,
        )
        try:
            delete_project_method = Image.automl_classification__delete_project(
                provider
            )
        except AttributeError:
            raise AttributeError("Could not import delete project phase.")

        feature_args["project_id"] = pytest.project_id

        delete_project_output = delete_project_method(**feature_args)
        original_response = delete_project_output.original_response
        standardized_response = delete_project_output.standardized_response

        assert isinstance(
            delete_project_output, ResponseType
        ), f"Expected ResponseType but got {type(delete_project_output)}"
        assert original_response is not None, "Original response should not be empty."
        assert isinstance(
            standardized_response, AutomlClassificationDeleteProjectDataClass
        ), f"Expected AutomlClassificationCreateProjectDataClass but got {type(standardized_response)}"
