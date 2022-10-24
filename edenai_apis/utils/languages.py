from collections import defaultdict
from importlib import import_module
from typing import Dict, Sequence, Union
import re
import pycountry
from langcodes import closest_supported_match, Language

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


def shorten_iso(iso):
    """Return the first part of a language iso (eg: 'en-US' -> 'en')"""
    if re.match("^[a-z][a-z]-[A-Z][A-Z]$", iso):
        return iso.split("-")[0]
    return iso


def check_language_format(iso_code):
    """Checks if language code name is formatted correctly (lang-extlang-Script-Reg)"""
    return bool(
        re.fullmatch(
            r"^[a-z]{2,3}(-[a-z]{2,3})?(-[A-Z][a-z]{3})?(-([A-Z]{2,3}|\d{3}))?", iso_code
        )
    )


def convert_three_two_letters(iso_code):
    """Converts an iso639-3 language code to an iso639-2 language code if possible"""
    if "-" in iso_code and len(iso_code.split("-")[0]) == 3:
        return str(Language.get(iso_code))
    if len(iso_code) != 3:
        return iso_code
    language = pycountry.languages.get(alpha_3=iso_code)
    if not language:  # can not find language in pycountry from it's alpha_3 code
        return str(Language.get(iso_code))
    return language.alpha_2 if hasattr(language, "alpha_2") else language.alpha_3


def load_language_constraints(provider_name, feature=None, subfeature=None):
    """Loads the list of languages supported by
    the provider for a couple of (feature, subfeature)"""
    info = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        provider_name=provider_name,
        feature=feature,
        subfeature=subfeature
    )
    default = defaultdict(lambda: None)
    languages = info.get("constraints", default).get("languages", [])
    return languages


def expend_laguages_for_user(list_languages):
    """Returns a list that extends the input list of languages (list_language)
    with formatted language tags and iso639-3 language codes"""
    appended_list = []
    for language in list_languages:
        if "-" in language:
            if (
                "Unknown language"
                not in Language.get(language.split("-")[0]).display_name()
            ):
                appended_list.append(convert_three_two_letters(language.split("-")[0]))
        appended_list.append(convert_three_two_letters(language))
    return appended_list


def load_standardized_language(feature, subfeature):
    """Displays a standardized list of languages for the pair (feature, subfeature)"""
    interface = import_module('edenai_apis.interface')
    providers = interface.list_providers(feature, subfeature)
    result = []
    for provider in providers:
        list_languages = expend_laguages_for_user(
            load_language_constraints(provider, feature, subfeature)
        )
        result = list(set(list_languages + result))
    return result


def load_standardized_language_for_providers(providers, feature, subfeature):
    """Displays a standardized list of languages for a list of providers
    for the pair (feature, subfeature)"""
    result = []
    for provider in providers:
        list_languages = expend_laguages_for_user(
            load_language_constraints(provider, feature, subfeature)
        )
        result = list(set(list_languages + result))
    return result


def format_language_name(language_name: str, isocode: str):
    """Formats language name by removing 'language'
    if this latter in Unknown or also removing 'region' if it's Unknown"""
    if "Unknown language" in language_name:
        formatted = re.search(r"\([a-zA-Z\s]*\)", language_name)
        return (
            f"Region: {formatted.group().split(')')[0].split('(')[1].strip()}"
            if formatted is not None
            else isocode
        )
    if "Unknown Region" in language_name:
        formatted = re.search(r"[a-zA-Z\s]*\(", language_name)
        return formatted.group().split("(")[0].strip()
    return language_name


def get_language_name_from_code(isocode):
    """Returns the language name from the isocode"""
    if "-" not in isocode:
        language = (
            pycountry.languages.get(alpha_2=isocode)
            if len(isocode) == 2
            else pycountry.languages.get(alpha_3=isocode)
        )
        output = Language.get(isocode).display_name() if not language else language.name
    else:
        language = Language.get(isocode)
        output = language.display_name()
    return format_language_name(output, isocode)


def provide_appropriate_language(iso_code, **kwds):
    if not check_language_format(iso_code):
        raise SyntaxError("Wrong format for language code name!!")
    if "provider_name" in kwds:
        list_languages: Sequence[str] = load_language_constraints(
            kwds.get("provider_name"), kwds.get("feature"), kwds.get("subfeature")
        )
    else:
        list_languages: Sequence[str] = kwds.get("list_languages").copy()
    while True:
        try:
            selected_code_language = closest_supported_match(iso_code, list_languages)
            break
        except RuntimeError:
            pass

    if "-" not in iso_code:
        return selected_code_language
    else:
        if selected_code_language is None:
            return None
        if (
            Language.get(iso_code).language
            == Language.get(selected_code_language).language
        ):
            if (
                Language.get(iso_code).region
                == Language.get(selected_code_language).region
            ):
                return selected_code_language
        return None


def update_provider_input_language(
    args: dict, provider_name: str, feature: str, subfeature: str
) -> Union[str, Dict]:
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
            return {
                "status": "fail",
                "provider": provider_name,
                "error": {
                    "is_provider_exception": True,
                    "message": str(exc),
                    "name": exc.__class__.__name__,
                    "code": getattr(exc, "code", None),
                },
            }

        # weither or not provider require a language
        accepts_null_language = (
            load_provider(
                ProviderDataEnum.PROVIDER_INFO,
                provider_name=provider_name,
                feature=feature,
                subfeature=subfeature,
            )
            .get("constraints", {})
            .get("allow_null_language", False)
        )

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
            return {
                "provider": provider_name,
                "status": "fail",
                "error": {
                    "is_provider_exception": True,
                    "message": f"language selected `{unsupported_language}` "
                    + "is not supported by the providers: `{provider_name}`",
                },
            }

        # change language argument for the conveted language
        if args.get("language"):
            args["language"] = supported_source_language
        elif args.get("source_language") and args.get("target_language"):
            args["source_language"] = supported_source_language
        if target_language:
            args["target_language"] = supported_target_language
    return "Done!"
