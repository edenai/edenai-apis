import json
import os
import pathlib
import time
from typing import Callable
from edenai_apis.utils.constraints import (
    validate_all_provider_constraints,
)

import pytest

from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.settings import outputs_path, features_path
from edenai_apis.utils.constraints import (
    validate_all_provider_constraints,
)
from edenai_apis.utils.types import AsyncLaunchJobResponseType

# TEXT_AUTOML_CLASSIFICATION = ["training_async", "prediction_async"]

MAX_TIME = 280
TIME_BETWEEN_CHECK = 10


def fake_cron_check(function: Callable, *args):
    """Fake cron function to use for test.
    Loop foop a certain number of time until we hit MAX_TIME
    or receive result from the providers' api.
    """
    time.sleep(5)
    current_time = MAX_TIME
    while current_time > 0:
        print(f"wait job result {MAX_TIME- current_time}s")
        api_output = function(*args)
        api_output = api_output.model_dump()
        if api_output["status"] != "pending":
            return api_output
        current_time = current_time - TIME_BETWEEN_CHECK
        time.sleep(TIME_BETWEEN_CHECK)


# skip test if no output in parameters
@pytest.mark.skipif("not config.getoption('output', skip=True)")
def test_outputs(provider, feature, subfeature, phase, generate=True):
    """
    Test output for a tuple provider,feature,subfeature
    and generate {subfeature}_{phase}_output.json file if generate = True
    """

    args = load_feature(
        FeatureDataEnum.SAMPLES_ARGS,
        feature=feature,
        subfeature=subfeature,
        phase=phase,
        provider_name=provider,
    )
    print("------------------------------------------------------------")
    args = validate_all_provider_constraints(provider, feature, subfeature, phase, args)
    
    if "async" in subfeature or "async" in phase:
        # Subfeature is asyncronous

        launch_job_response: AsyncLaunchJobResponseType = load_provider(
            ProviderDataEnum.SUBFEATURE,
            provider_name=provider,
            subfeature=subfeature,
            feature=feature,
            phase=phase,
            suffix="__launch_job",
        )(**args)

        # Check if the subfeature implements a __get_job_result method
        get_job_result = load_provider(
            ProviderDataEnum.SUBFEATURE,
            provider_name=provider,
            feature=feature,
            subfeature=subfeature,
            phase=phase,
            suffix="__get_job_result",
        )

        # loop over __get_job_result function to get async subfeature result
        api_output = fake_cron_check(
            get_job_result, launch_job_response.provider_job_id
        )

    else:
        # Subfeature is not asyncronous
        api_output = load_provider(
            ProviderDataEnum.SUBFEATURE,
            provider_name=provider,
            subfeature=subfeature,
            feature=feature,
            phase=phase,
            args=args,
        )(**args).model_dump()

    # Generate output
    if generate:
        output_path = os.path.join(outputs_path(provider), feature)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        if phase:
            json_output_path = os.path.join(
                output_path, f"{subfeature}_{phase}_output.json"
            )
        else:
            json_output_path = os.path.join(
                output_path, f"{subfeature}{phase}_output.json"
            )
        print(f"Write {json_output_path}.")
        with open(json_output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(api_output, indent=2, default=str))
        if api_output.get("standardized_response"):
            feature_sample_path = os.path.join(
                features_path, feature, subfeature, subfeature + "_response.json"
            )
            if not os.path.isfile(feature_sample_path):
                print(f"Write {feature_sample_path}.")
                with open(feature_sample_path, "w", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            api_output["standardized_response"], indent=2, default=str
                        )
                    )

    return api_output
