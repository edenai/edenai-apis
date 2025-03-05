from pprint import pprint
from time import sleep
from typing import Tuple

import requests

from edenai_apis.interface import compute_output, get_async_job_result
from edenai_apis.loaders.data_loader import FeatureDataEnum
from edenai_apis.loaders.loaders import load_feature


def process_async_get_result(
    provider: str,
    feature: str,
    subfeature: str,
    phase: str,
    async_job_id: str,
    max_time=300,
    sleep_time=5,
):
    while max_time > 0:
        res = get_async_job_result(
            provider_name=provider,
            feature=feature,
            subfeature=subfeature,
            async_job_id=async_job_id,
            phase=phase,
        )
        if res["status"] == "failed":
            raise Exception(res["error"])
        elif res["status"] == "succeeded":
            return
        sleep(sleep_time)
        max_time -= sleep_time

    raise TimeoutError(f"Async job timed out after {max_time} seconds")


def process_provider(provider_info: Tuple[str, str, str, str]):
    provider, feature, subfeature, phase = provider_info
    if phase == "create_project":
        return None
    try:
        arguments = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature=feature,
            subfeature=subfeature,
            phase=phase,
        )
    except NotImplementedError:
        return None
    try:
        res = compute_output(
            provider_name=provider,
            feature=feature,
            subfeature=subfeature,
            args=arguments,
            phase=phase,
        )
        if res["status"] == "fail":
            raise Exception(res["error"])

        # poll for result if async job
        if "provider_job_id" in res:
            process_async_get_result(
                provider=provider,
                feature=feature,
                subfeature=subfeature,
                phase=phase,
                async_job_id=res["provider_job_id"],
            )

        return (provider, feature, subfeature, None)
    except Exception as exc:
        return (provider, feature, subfeature, exc)


def fetch_provider_subfeatures():
    url = "https://api.edenai.run/v2/info/provider_subfeatures"
    response = requests.get(url)
    return response.json()


def main():
    not_working = []
    provider_subfeatures = fetch_provider_subfeatures()
    all_providers = [
        (
            provider["provider"]["name"],
            provider["feature"]["name"],
            provider["subfeature"]["name"],
            provider.get("phase") or "",
        )
        for provider in provider_subfeatures
    ]

    for provider_info in all_providers:
        print("test: ", provider_info)
        result = process_provider(provider_info)
        if result is None:
            continue
        provider, feature, subfeature, error = result
        if error is not None:
            print("ERROR: ", result)
            not_working.append((provider, feature, subfeature, error))

    print("=================================")
    print("NOT WORKING PROVIDERS WITH ERRORS")
    pprint(not_working)
    print("=================================")

    if not_working:
        raise Exception


if __name__ == "__main__":
    main()
