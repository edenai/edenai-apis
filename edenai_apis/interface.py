# pylint: disable=locally-disabled, too-many-branches
import inspect
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union, overload
from uuid import uuid4

from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.utils.languages import update_provider_input_language
from edenai_apis.utils.compare import assert_equivalent_dict
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import AsyncLaunchJobResponseType

ProviderDict = Dict[
    str, Dict[
        str, Dict[
            str, Union[
                Dict[
                    str, Literal[True]
                ],
                Literal[True]
            ]
        ]
    ]
]
FeatureSubfeatureProviderTuple = Union[Tuple[str, str, str], Tuple[str, str, str, str]]
ProviderList = List[FeatureSubfeatureProviderTuple]


@overload
def list_features(
    provider_name: Optional[str] = None,
    feature: Optional[str] = None,
    subfeature: Optional[str] = None,
    as_dict: Literal[False] = False,
) -> ProviderList: ...
@overload
def list_features(
    provider_name: Optional[str] = None,
    feature: Optional[str] = None,
    subfeature: Optional[str] = None,
    as_dict: Literal[True] = True,
) -> ProviderDict: ...
def list_features(
    provider_name: Optional[str] = None,
    feature: Optional[str] = None,
    subfeature: Optional[str] = None,
    as_dict: bool = False,
) -> Union[ProviderList, ProviderDict]:
    """
    Gets all possible provider/feature/subfeature or provider/feature/subfeature/phase as a list if `as_dict` is `False`,
    otherwise returns the resutls as a dict.

    Example of dict result:
    ```python
    {
        [provider]: {
            [feature]: {
                [subfeature]: True
            }
        }
    }
    ```
    or if phase is present
    ```python
    {
        [provider]: {
            [feature]: {
                [subfeature]: {
                    [phase] : True
                }
            }
        }
    }
    ```

    Examples list result:
    ```python
    [(provider, feature, subfeature, phase), (provider, feature, subfeature, phase), ...]
    ```
    or
    ```python
    [(provider, feature, subfeature), (provider, feature, subfeature), ...]
    ```

    Args:
        provider_name (str, optional): Eden AI provider name. Defaults to None.
        feature (str, optional): Eden AI feature name. Defaults to None.
        subfeature (str, optional): Eden AI subfeature name. Defaults to None.
        as_dict (bool, optional): Whether or not return the results as a Dict. Defaults to False.

    Returns:
        (list | dict): Return all possible provider/feature/subfeature or provider/feature/subfeature/phase as a list or dict
    """

    method_set: Set[FeatureSubfeatureProviderTuple] = set()
    ApiClasses: List[Type[ProviderApi]] = load_provider(ProviderDataEnum.CLASS)
    for cls in ApiClasses:
        if (
            not provider_name or cls.provider_name == provider_name
        ):  # filter for provider_name if provided

            # detect feature,subfeature,phase by looking at methods names
            for method_name in [
                fct
                for fct in cls.__dict__.keys()
                if inspect.isfunction(getattr(cls, fct))
            ]:
                if not method_name.startswith("_") and "__" in method_name:
                    feature_i, subfeature_i, *others = method_name.split("__")
                    if (
                        not (feature or feature == feature_i)
                        and not (subfeature or subfeature == subfeature_i)
                        ):  # filter by subfeature if provided
                            if len(others) > 0 and "async" not in subfeature_i:
                                phase = others[0]
                                method_set.add(
                                    (cls.provider_name, feature_i, subfeature_i, phase)
                                )
                            else:
                                method_set.add(
                                    (cls.provider_name, feature_i, subfeature_i)
                                )
    method_list: ProviderList = list(method_set)
    method_list.sort()
    if not as_dict:
        return method_list  # return a list

    # return resutls as dict
    result = {}
    for provider, feature_i, subfeature_i, *phase in method_list:
        if provider not in result:
            result[provider] = {}
        if feature_i not in result[provider]:
            result[provider][feature_i] = {}
        have_phases = isinstance(result[provider][feature_i].get(subfeature_i), dict)
        if result[provider][feature_i].get(subfeature_i) is None or have_phases:
            if phase:
                if not have_phases:
                    result[provider][feature_i][subfeature_i] = {}
                if not phase[0] in result[provider][feature_i][subfeature_i]:
                    result[provider][feature_i][subfeature_i][phase[0]] = True
            else:
                result[provider][feature_i][subfeature_i] = True

    return result

def list_providers(
    feature: Optional[str] = None, subfeature: Optional[str] = None
) -> List[str]:
    """
    Lists all providers that implement the given feature/subfeature

    Args:
        feature(str, optional): Edenai AI feature name. Default to `None`.
        subfeature(str, optional): Edenai AI subfeature name. Default to `None`.

    Returns:
        List[str]: list of provider names
    """
    # TO DO: keep the set as output, don't convert to list
    providers_set = set()
    for provider, feature_i, subfeature_i, *_phase in list_features():
        if not feature or feature_i == feature:
            if not subfeature or subfeature_i == subfeature:
                providers_set.add(provider)
    return list(providers_set)


def compute_output(
    provider_name: str,
    feature: str,
    subfeature: str,
    args: Dict[str, Any],
    phase: str = "",
    fake: bool = False,
) -> Dict:
    """
    Compute subfeature for provider and subfeature

    Args:
        provider_name (str): EdenAI provider name
        feature (str): EdenAI feature name
        subfeature (str): EdenAI subfeature name
        phase (str): Eden AI phase name if give, Default to `Literal[""]`
        args (Dict): inputs arguments for the feature call
        fake (bool, optional): take result from sample. Defaults to `False`.

    Returns:
        dict: Result dict
    """
    # check if the function we're running is asyncronous
    is_async = ("_async" in phase) if phase else ("_async" in subfeature)
    # suffix is used for async
    suffix = "__launch_job" if is_async else ""

    status = "success"

    if fake:
        # Return mocked results
        if is_async:
            subfeature_result: Any = AsyncLaunchJobResponseType(provider_job_id=str(uuid4())).dict()
        else:
            if phase in ["upload_image", "delete_image"]:
                subfeature_result = {"status": "success"}
            else:
                sample_args = load_feature(
                    FeatureDataEnum.SAMPLES_ARGS,
                    feature=feature,
                    subfeature=subfeature,
                    phase=phase,
                )

                # Check if the right arguments were sent by checking
                # if they are equivalent to samples arguments
                assert_equivalent_dict(sample_args, args)
                output = load_provider(
                    ProviderDataEnum.OUTPUT,
                    provider_name=provider_name,
                    feature=feature,
                    subfeature=subfeature,
                    phase=phase,
                )
                subfeature_result = {
                    "original_response": output["original_response"],
                    "standarized_response": output["standarized_response"],
                }

    else:
        # Fake == False : Compute real output
        language_output = update_provider_input_language(
            args, provider_name, feature, subfeature
        )
        # means that an error was found
        if isinstance(language_output, dict):
            return language_output

        try:
            subfeature_result = load_provider(
                ProviderDataEnum.SUBFEATURE,
                provider_name=provider_name,
                feature=feature,
                subfeature=subfeature,
                phase=phase,
                suffix=suffix,
            )(**args).dict()

        except ProviderException as exception:
            subfeature_result = {
                "error": {
                    "message": str(exception),
                    "code": getattr(exception, "code", None),
                }
            }

            status = "fail"

    final_result: Dict[str, Any] = {
        "status": status,
        "provider": provider_name,
        **subfeature_result
    }

    return final_result

def check_provider_constraints(
    provider_name: str,
    feature: str,
    subfeature: str,
    phase: Optional[str] = None,
    constraints: Optional[Dict] = None,
) -> Tuple[bool, str]:
    """
    Checks if provider is ok for one feature and subfeature (and phase if given)
    Check for constraints as well

    Args:
        provider_name (str): EdenAI provider name
        feature (str): only feature key.
        subfeature (str): only subfeature key.
        phase (str, optional): EdenAI phase. Defaults to `None`.
        constraints (Dict): constraints. Defaults to `None`.

    Returns:
        Tuple[bool, str]: Provider is ok, debug string
    """

    # Refacto : use list_providers ?
    subfeatures_providers = list_features(as_dict=True)
    provider_info = subfeatures_providers.get(provider_name, None)
    if not provider_info:
        return False, f"Provider : '{provider_name}' unknown."
    if feature not in provider_info:
        return (
            False,
            f"Provider : '{provider_name}' does not provide an API for '{feature} {subfeature}'",
        )
    if subfeature not in provider_info[feature]:
        return (
            False,
            f"Provider : '{provider_name}' does not provide an API for '{feature} {subfeature}'",
        )
    if phase:
        if not isinstance(provider_info[feature][subfeature], dict) or (
            isinstance(provider_info[feature][subfeature], dict)
            and phase not in provider_info[feature][subfeature].keys()
        ):
            return (
                False,
                f"Provider : '{provider_name}' does not provide an API for "
                + "'{feature} {subfeature} {phase}'",
            )

    if constraints:
        for key, values in constraints:
            for value in values:
                if (
                    not value
                    in load_provider(
                        ProviderDataEnum.PROVIDER_INFO,
                        provider_name=provider_name,
                        feature=feature,
                        subfeature=subfeature,
                        phase=phase,
                    )[key]
                ):
                    return (
                        False,
                        f"constraint : '{key} - {values}' not filled for '{provider_name}'",
                    )
    return True, "All Good!"

def get_async_job_result(
    provider_name: str,
    feature: str,
    subfeature: str,
    async_job_id: AsyncLaunchJobResponseType,
    phase: str = "",
    fake: bool = False,
) -> Dict:
    """Get async result from job id

    Args:
        provider_name (str): EdenAI provider name
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature
        async_job_id (str): async job id to get result to
        phase (str): EdenAI phase. Default to empty string ("")
        fake (bool): Load fake results

    Returns:
        Dict: Result dict
    """

    if fake is True:
        # Load fake data from edenai_apis' saved output
        fake_result = load_provider(
            ProviderDataEnum.OUTPUT,
            provider_name=provider_name,
            feature=feature,
            subfeature=subfeature,
            phase=phase,
        )
        fake_result["provider_job_id"] = async_job_id

        return fake_result

    try:
        subfeature_result = load_provider(
            ProviderDataEnum.SUBFEATURE,
            provider_name=provider_name,
            subfeature=subfeature,
            feature=feature,
            phase=phase,
            suffix="__get_job_result",
        )(async_job_id).dict()

    except ProviderException as exception:
        subfeature_result = {
            "status": "failed",
            "provider": provider_name,
            "provider_job_id": async_job_id,
            "error": {
                "message": str(exception),
                "code": getattr(exception, "code", None),
            },
        }
    return subfeature_result

def get_async_job_webhook_result(
    provider_name: str, feature: str, subfeature: str, data: Dict, phase: str = ""
) -> Optional[Dict]:
    """Format result from webhook to standarized response

    Args:
        provider_name (str): EdenAI provider name
        feature (str): EdenAI feature
        subfeature (str): EdenAI subfeature
        data (Dict): body from webhook

    Returns:
        Dict: Result dict
    """
    try:
        subfeature_result = load_provider(
            ProviderDataEnum.SUBFEATURE,
            provider_name=provider_name,
            subfeature=subfeature,
            feature=feature,
            phase=phase,
            suffix="__get_results_from_webhook",
        )(data).dict()
        return subfeature_result
    except AttributeError:
        pass
    except Exception as exception:
        is_provider_exception = isinstance(exception, ProviderException)
        subfeature_result = {
            "status": "failed",
            "provider": provider_name,
            "error": {
                "is_provider_exception": is_provider_exception,
                "message": str(exception),
                "name": exception.__class__.__name__,
                "code": getattr(exception, "code", None),
            },
        }
        return subfeature_result
