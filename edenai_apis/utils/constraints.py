import io
import mimetypes
from typing import Dict, List, Optional, Union

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import provide_appropriate_language


def validate_all_provider_constraints(
    provider: str, feature: str, subfeature: str, args: dict
) -> dict:
    """
    Validate inputs arguments against provider constraints

    Args:
        - provider (str): provider name
        - feature (str): feature name
        - subfeature (str): subfeature name
        - args (dict): dictionnary of input arguments

    Returns:
        - args: updated/validated args
    """

    # load provider constraints
    provider_constraints = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        provider_name=provider,
        feature=feature,
        subfeature=subfeature,
    ).get("constraints")

    if provider_constraints is not None:
        validated_args = args.copy()
        ## Validate here

        # file types
        validated_args = validate_input_file_type(
            provider_constraints, validated_args, provider
        )

        # languages
        validated_args = validate_input_language(
            provider_constraints, validated_args, provider, feature, subfeature
        )

        # ...

        return validated_args

    return args


def validate_input_file_type(constraints: dict, args: dict, provider: str) -> dict:
    """Check that a provider offers support for the input file type

    Args:
        - constraints (dict): all constraints (on inputs) of the provider
        - args (dict): inputs passed to the provider call
        - provider (str): provider name

    Returns:
        - `args` (dict): same or updated args

    Raises:
        - `ProviderException`: if file is not supported
    """
    provider_file_type_constraints: List[str] = constraints.get("file_types", [])

    input_file: Optional[io.BufferedReader] = args.get("file")

    if input_file and len(provider_file_type_constraints) > 0:
        input_file_type, _ = mimetypes.guess_type(input_file.name)

        if input_file_type is None:
            # if mimetype is not recognized we don't validate it
            # eg: webp and raw images are not recognized but are accepted by google ocr
            return args

        # constraint can be written as "image/*" for example
        # it means it accepts all types of images
        type_glob = [
            constraint.split("/")[0]
            for constraint in provider_file_type_constraints
            if constraint.endswith("*")
        ]

        if input_file_type not in provider_file_type_constraints and not any(
            [global_type in input_file_type for global_type in type_glob]
        ):
            raise ProviderException(
                f"Provider {provider} doesn't support file type: {input_file_type} "
                f"for this feature. \n"
                f"Supported types are {provider_file_type_constraints}"
            )

    return args


def validate_input_language(
    constraints: dict, args: dict, provider_name: str, feature: str, subfeature: str
) -> Dict:
    """Updates the args input to provide the appropriate language
    supported by the provider from user input language,
    while checking at the same time if None language is acceptable
    Args:
        args (dict): feature arguments, exp: {text: 'alleluia', language='en'} \n
        provider_name (str): the provider name, exp: Google\n
        feature (str): the feature name, exp: text\n
        subfeature (str): the subfeature name, exp: sentiment_analysis\n
    Returns:
        Union[str,Dict]: return a dict representing an error if the language or
        source_language is None and the provider do not accept None values,
        or if the target_language is None. Otherwise, returns a string 'Done!'
    """
    # get languages
    source_language = args.get("language") or args.get("source_language")
    target_language = args.get("target_language")
    if source_language:
        try:
            # convert languages, return None if not supported
            supported_source_language: Union[str, None] = provide_appropriate_language(
                source_language,
                provider_name=provider_name,
                feature=feature,
                subfeature=subfeature,
            )
            if target_language:
                supported_target_language: Union[
                    str, None
                ] = provide_appropriate_language(
                    target_language,
                    provider_name=provider_name,
                    feature=feature,
                    subfeature=subfeature,
                )
        except SyntaxError as exc:
            raise ProviderException(str(exc))

        # weither or not provider require a language
        accepts_null_language = constraints.get("allow_null_language", False)

        # if source language is not supported or
        # if there is a target language and it is not supported
        if (supported_source_language is None and not accepts_null_language) or (
            target_language and supported_target_language is None
        ):
            unsupported_language = (
                source_language
                if supported_source_language is None
                else target_language
            )
            raise ProviderException(
                f"language selected `{unsupported_language}` is not supported by"
                f" the providers: `{provider_name}`"
            )

        # change language argument for the conveted language
        if args.get("language"):
            args["language"] = supported_source_language
        elif args.get("source_language") and args.get("target_language"):
            args["source_language"] = supported_source_language
        if target_language:
            args["target_language"] = supported_target_language
    return args
