import io
import mimetypes
from typing import Dict, List, Optional

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.languages import (
    LanguageErrorMessage,
    provide_appropriate_language,
    load_standardized_language
)

def validate_input_file_type(constraints: dict, provider: str, args: dict) -> dict:
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
            supported_types = ",".join(provider_file_type_constraints)
            raise ProviderException(
                f"Provider {provider} doesn't support file type: {input_file_type} "
                f"for this feature. "
                f"Supported mimetypes are {supported_types}"
            )
    return args


def validate_single_language(
    provider_name: str,
    feature,
    subfeature,
    language: dict,
    null_language_accepted: bool,
) -> Optional[str]:
    """
    Validate and format given language

    Args:
        - provider_name (str)
        - feature (str): feature name
        - subfeature (str): subfeature name
        - language (dict): Dictionnary with `key` and `value` of input language. `value` canbe None. (ex: { 'key': 'source_langue', 'value': 'en'})
        - null_language_accepted (bool): if Provider can auto-detect langauages (accepts providing None as language)

    Returns:
        - language (str | None) validated language, can be None if provider accepts it

    Raises:
        - `ProviderException`: if language is not supported or cannot be None
    """

    # if user specifies the auto-detect language
    if language['value'] and language['value'].lower() == "auto-detect":
        language['value'] = None

    if not language['value']:
        if null_language_accepted is True:
            return language['value']
        else:
            raise ProviderException(LanguageErrorMessage.LANGUAGE_REQUIRED(language['key']))

    try:
        formated_language = provide_appropriate_language(
            language['value'],
            provider_name=provider_name,
            feature=feature,
            subfeature=subfeature,
        )
    except SyntaxError as exc:
        raise ProviderException(LanguageErrorMessage.LANGUAGE_SYNTAX_ERROR(language['value']))

    if null_language_accepted is False:
        if formated_language is None:
            if "-" in language['value']:
                supported_languages = load_standardized_language(feature, subfeature, [provider_name])
                suggested_language = language['value'].split("-")[0]
                if suggested_language in supported_languages:
                    raise ProviderException(
                        LanguageErrorMessage.LANGUAGE_GENERIQUE_REQUESTED(language['value'], suggested_language, language['key'])
                    )
            raise ProviderException(
                LanguageErrorMessage.LANGUAGE_NOT_SUPPORTED(language['value'], language['key'])
            )

    return formated_language


def validate_all_input_languages(
    constraints: dict, args: dict, provider_name: str, feature: str, subfeature: str
) -> Dict:
    """
    Updates the args input to provide the appropriate language
    supported by the provider from user input language,
    while checking at the same time if None language is acceptable

    Args:
        - constraints: provider constraints for this subfeature
        - args (dict): feature arguments, exp: {text: 'alleluia', language='en'}
        - provider_name (str): the provider name, exp: Google
        - feature (str): the feature name, exp: text
        - subfeature (str): the subfeature name, exp: sentiment_analysis

    Returns:
        - dict: updated args
    """

    accepts_null_language = constraints.get("allow_null_language", False)

    for argument_name, argument_value in args.items():
        if 'language' not in argument_name:
            continue

        args[argument_name] = validate_single_language(
            provider_name=provider_name,
            feature=feature,
            subfeature=subfeature,
            language={ 'key': argument_name, 'value': argument_value },
            null_language_accepted=accepts_null_language,
        )
    return args


def validate_all_provider_constraints(
    provider: str, feature: str, subfeature: str, phase: str, args: dict
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
    provider_info = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        provider_name=provider,
        feature=feature,
        subfeature=subfeature,
        phase=phase,
    )
    provider_constraints = provider_info.get("constraints")


    if provider_constraints is not None:
        validated_args = args.copy()
        ## Validate here

        # file types
        validated_args = validate_input_file_type(
            provider_constraints, provider, validated_args
        )

        # languages
        validated_args = validate_all_input_languages(
            provider_constraints, validated_args, provider, feature, subfeature
        )

        # ...

        return validated_args

    return args