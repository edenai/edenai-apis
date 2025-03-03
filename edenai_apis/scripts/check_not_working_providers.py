import sys
from pprint import pprint

import requests
from edenai_apis.interface import compute_output
from edenai_apis.loaders.data_loader import FeatureDataEnum
from edenai_apis.loaders.loaders import load_feature

HOURLY = "hourly"

if __name__ == "__main__":
    interval = sys.argv[1]
    not_working = []
    query_is_working = "?is_working=False" if interval == HOURLY else ""
    provider_subfeatures = requests.get(
        url=f"https://api.edenai.run/v2/info/provider_subfeatures{query_is_working}"
    ).json()
    all_providers = [
        (
            provider["provider"]["name"],
            provider["feature"]["name"],
            provider["subfeature"]["name"],
            provider.get("phase", ""),
        )
        for provider in provider_subfeatures
    ]
    for provider, feature, subfeature, phase in all_providers:
        if phase == "create_project":
            continue
        try:
            arguments = load_feature(
                FeatureDataEnum.SAMPLES_ARGS,
                feature=feature,
                subfeature=subfeature,
                phase=phase,
            )
        except NotImplementedError:
            continue
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

        except Exception as exc:
            print(provider, feature, subfeature)
            print(exc)
            not_working.append((provider, feature, subfeature, exc))

    print("=================================")
    print("NOT WORKING PROVIDERS WITH ERRORS")
    pprint(not_working)
    print("=================================")

    if not_working:
        raise Exception
