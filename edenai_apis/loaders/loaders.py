# pylint: disable=locally-disabled, unused-argument
"""
Module that contains loaders function that returns various types data
from features & provider
data function are defined in `edenai_apis.loaders.data_loaders`
"""
import inspect
from importlib import import_module
from typing import Optional

from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum


def load_feature(
    data_feature: FeatureDataEnum,
    provider_name: str = "",
    feature: str = "",
    subfeature: str = "",
    phase: str = "",
    suffix: str = "",
    **kwargs,
):
    """
    Call data_loader functions and returns data related to features
    availables functions (found in data_loaders.py) are:
        `load_dataclass(feature: str, subfeature: str, phase: str = None) -> BaseModel`
        `load_samples(
            feature: str,
            subfeature: str,
            phase: str = "",
            provider_name: str = "",
            suffix: str = ""
            ) -> Dict:`

    Args:
        data_feature(FeatureDataEnum): wich data_loader to calll
        feature (str): Feature name. Default to `""`.
        subfeature (str): subFeature name. Default to `""`.
        phase (str): phase name. Default to `""`.
        suffix (str): suffix name. Default to `""`.
        **kwarg: additional arguments that will be passed to the
            data_loader function. Any useless kwargs will be ignored

    Returns:
        - Any: The returned value of data_loader function
    """
    args = locals()
    if args.get("kwargs") is not None and len(args["kwargs"]) != 0:
        more_args_dict: dict = args["kwargs"]
        for key, val in more_args_dict.items():
            args[key] = val
    del args["kwargs"]

    load_data_module = import_module("edenai_apis.loaders.data_loader")
    load_data_function = getattr(load_data_module, data_feature.value)

    args_data_function = list(inspect.signature(load_data_function).parameters.keys())
    args_to_use = {key: val for key, val in args.items() if key in args_data_function}

    return load_data_function(**args_to_use)


def load_provider(
    data_provider: ProviderDataEnum,
    provider_name: str = "",
    feature: str = "",
    subfeature: str = "",
    phase: Optional[str] = "",
    suffix: str = "",
    **kwargs,
):
    """
    Call data_loader functions and returns data related to providers
    availables functions (found in data_loaders.py) are:
        `load_info_file(provider_name: str = "") -> Dict`
        `load_class(provider_name: Optional[str] = None) -> Tuple[List[ProviderInterface], ProviderInterface]`
        `load_output(provider_name: str, feature: str, subfeature: str, phase: str = "") -> Dict`
        `load_subfeature(provider_name: str, feature: str, subfeature: str, phase: str = "", suffix="") -> Callable`
        `load_provider_subfeature_info(provider_name: str, feature: str, subfeature: str, phase: str = "")`
        `load_key(provider_name, location=False)`

    Args:
        data_feature(ProviderDataEnum): wich data_loader to call
        feature (str): Feature name. Default to `""`.
        subfeature (str): subFeature name. Default to `""`.
        phase (str): phase name. Default to `""`.
        suffix (str): suffix name. Default to `""`.
        **kwarg: additional arguments that will be passed to the
            data_loader function (eg: `location` can be passed to `load_key`).
            Any useless kwargs will be ignored.

    Returns:
        - Any: The returned value of data_loader function
    """
    args = locals()
    if args.get("kwargs") is not None and len(args["kwargs"]) != 0:
        more_args_dict: dict = args["kwargs"]
        for key, val in more_args_dict.items():
            args[key] = val
    del args["kwargs"]

    load_data_module = import_module("edenai_apis.loaders.data_loader")
    load_data_function = getattr(load_data_module, data_provider.value)

    args_data_function = list(inspect.signature(load_data_function).parameters.keys())
    args_to_use = {key: val for key, val in args.items() if key in args_data_function}

    return load_data_function(**args_to_use)
