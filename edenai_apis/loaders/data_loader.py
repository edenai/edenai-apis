import os
from enum import Enum
from importlib import import_module
from typing import Callable, Dict, List, Optional, Union, overload, Type

from pydantic import BaseModel

from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.loaders.utils import load_json, check_messsing_keys
from edenai_apis.settings import info_path, keys_path, outputs_path
from edenai_apis.utils.compare import is_valid


class FeatureDataEnum(Enum):
    DATA_CLASS = "load_dataclass"
    SAMPLES_ARGS = "load_samples"


class ProviderDataEnum(Enum):
    INFO_FILE = "load_info_file"
    CLASS = "load_class"
    OUTPUT = "load_output"
    SUBFEATURE = "load_subfeature"
    PROVIDER_INFO = "load_provider_subfeature_info"
    KEY = "load_key"


def load_key(provider_name, location=False, api_keys: Dict = {}):
    """Get settings for a provider name of from passed apik_keys dict

    Args:
        provider_name (str): EdenAI provider name
        location (bool, optional): Return as tuple
        and path is second. Defaults to False.
        apik_keys (Dict): users api keys to be used instead of default settings keys
    """

    provider_settings_path = os.path.join(keys_path, provider_name + "_settings.json")
    provider_settings_data = load_json(provider_settings_path)
    data = provider_settings_data

    if api_keys:
        data = api_keys
        # check_messsing_keys(provider_settings_data, data)

    if location:
        return data, provider_settings_path if not api_keys else ""
    return data


@overload
def load_class() -> List[Type[ProviderInterface]]: ...


@overload
def load_class(provider_name: str) -> Type[ProviderInterface]: ...


def load_class(
    provider_name: Optional[str] = None,
) -> Union[List[Type[ProviderInterface]], Type[ProviderInterface]]:
    """Get all ProviderInterface in providers package

    Args:
        provider_name (str, optional): get only Provider
        that match provider_name. Defaults to None.

    Raises:
        ValueError: if provider_name is wrong

    Returns:
        Union[List[ProviderInterface], ProviderInterface]: returnd ProviderInterface class(es)
            single class if provider_name is provided, or a list if provider_name is None.
    """
    from edenai_apis import apis

    api_class_list: List[Type[ProviderInterface]] = [
        getattr(apis, api) for api in dir(apis) if is_valid(".*Api", api)
    ]
    api_class_list.sort(key=lambda api: api.provider_name)
    if provider_name:
        # get first occurence of class with given provider_name, or None
        api_class: Optional[Type[ProviderInterface]] = next(
            filter(lambda cls: cls.provider_name == provider_name, api_class_list), None
        )

        if api_class is None:
            raise ValueError(
                f"No ProviderInterface class implemented for provider: {provider_name}."
            )

        return api_class
    return api_class_list


def load_dataclass(
    feature: str, subfeature: str, phase: Optional[str] = None
) -> BaseModel:
    """Get OutputDataClass for one feature and one subfeature

    Args:
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature

    Returns:
        OutputDataClass: dataclass related to subfeature
    """
    module_path = f"edenai_apis.features.{feature}.{subfeature}.{subfeature}_dataclass"
    dataclass_name = subfeature.replace("_", " ").title().replace(" ", "") + "DataClass"
    if (
        phase and phase != "launch_similarity"
    ):  # if there is a phase, there will be a dataclass by phase
        dataclass_name = (
            f"{subfeature}_{phase}".replace("_", " ").title().replace(" ", "")
            + "DataClass"
        )
        module_path = (
            f"edenai_apis.features.{feature}.{subfeature}"
            + f".{phase}.{subfeature}_{phase}_dataclass"
        )
    dataclass_module = import_module(module_path)
    return getattr(dataclass_module, dataclass_name)


def load_info_file(provider_name: str = "") -> Dict:
    """Compile and return all info.json content in a dictionnary
    if no provider name is specified, otherwise, load info.json for
    a specific provider

    Args:
        provider_name (str): EdenAI provider name. Default to empty stirng

    Returns:
        a dict containing the info.json information if the provider name is specified,\n
        otherwise return Dict[Tuple[str,str,str], Dict] mapping a tuple key with the format
        (provider_name, feature, subfeature) to it's info.json information

    """

    if provider_name:
        return load_json(info_path(provider_name))

    all_infos = {}
    for provider_name_i in map(
        lambda provider_class: provider_class.provider_name, load_class()
    ):
        provider_info = load_info_file(provider_name_i)
        for feature in provider_info:
            if feature == "_metadata":
                all_infos[(provider_name_i, feature, "privacy_url")] = provider_info[
                    feature
                ]["privacy_url"]
                continue
            for subfeature in provider_info[feature]:
                if (
                    not provider_info.get(feature, {})
                    .get(subfeature, {})
                    .get("version")
                ):
                    for phase in provider_info.get(feature, {}).get(subfeature, []):
                        all_infos[(provider_name_i, feature, subfeature, phase)] = (
                            provider_info[feature][subfeature][phase]
                        )
                else:
                    all_infos[(provider_name_i, feature, subfeature)] = provider_info[
                        feature
                    ][subfeature]
    return all_infos


global ALL_PROVIDERS_INFOS
ALL_PROVIDERS_INFOS = load_info_file()


def load_provider_subfeature_info(
    provider_name: str, feature: str, subfeature: str, phase: str = ""
):
    """Get provider subfeature info.json from memory"""
    global ALL_PROVIDERS_INFOS
    if len(ALL_PROVIDERS_INFOS) == 0:
        ALL_PROVIDERS_INFOS = load_info_file()
    if phase:
        return ALL_PROVIDERS_INFOS[(provider_name, feature, subfeature, phase)].copy()
    return ALL_PROVIDERS_INFOS.get((provider_name, feature, subfeature), {}).copy()


def load_output(
    provider_name: str, feature: str, subfeature: str, phase: str = ""
) -> Dict:
    """Load versionned output of provider for a subfeature

    Args:
        provider_name (str): EdenAI provider name
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature
        phase (str): EdenAI phase for automl. Default to ''.

    Returns:
        Dict: output
    """
    if phase:
        path = os.path.join(
            outputs_path(provider_name), feature, f"{subfeature}_{phase}_output.json"
        )
    else:
        path = os.path.join(
            outputs_path(provider_name), feature, f"{subfeature}_output.json"
        )
    data = load_json(path)
    return data


def load_subfeature(
    provider_name: str, feature: str, subfeature: str, phase: str = "", suffix: str = ""
) -> Callable:
    """Load subfeature method for provider

    Args:
        provider_name (str): EdenAI provider name
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature
        phase (str) : EdenAI phase for automl. Default to ''.
        suffix (str, optional): Suffix to add at the end (mainly for async job). Defaults to ''.

    Returns:
        Callable: function to get subfeature result
    """
    api = load_class(provider_name)()
    phase_ = f"__{phase}" if phase else ""

    return getattr(api, f"{feature}__{subfeature}{phase_}{suffix}")


def load_samples(
    feature: str, subfeature: str, phase: str = "", provider_name: Optional[str] = None
) -> Dict:
    """Get arguments for the pair (feature, subfeature)
    or for the triple (feature, subfeature, phase)

    Args:
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature
        phase (str): EdenAI phase. Default to ''

    Returns:
        Dict: arguments related to a subfeautre or phase
    """
    normalized_subfeature = f"{subfeature}{f'_{phase}' if phase else ''}"
    imp = import_module(
        f"edenai_apis.features.{feature}.{subfeature}{f'.{phase}' if phase else ''}.{normalized_subfeature}_args"
    )
    return getattr(imp, f"{normalized_subfeature}_arguments")(provider_name)
