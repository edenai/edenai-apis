from typing import Dict
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


def call_subfeature(
    provider_name: str,
    feature: str,
    subfeature: str,
    args: dict,
    phase: str = "",
    suffix: str = "",
) -> Dict:
    """Call subfeature method for provider and register it to db history

    Args:
        provider_name (str): EdenAI provider name
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature
        args (Dict): call arguments
        phase (str): EdenAI phasen name. Default to ''
        suffix (str, optional): Suffix to add at the end (mainly for async job). Defaults to ''.

    Returns:
        Dict: calls and returns the subfeature result
    """
    result = load_provider(
        ProviderDataEnum.SUBFEATURE,
        provider_name=provider_name,
        feature=feature,
        subfeature=subfeature,
        phase=phase,
        suffix=suffix,
    )(**args)

    return result
