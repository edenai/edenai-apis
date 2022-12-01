import re
from collections import defaultdict
from importlib import import_module
from typing import List, Optional, Sequence

import pycountry
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from langcodes import Language, closest_supported_match


class LanguageErrorMessage:
    LANGUAGE_REQUIRED = (
        "This provider doesn't auto-detect languages, "
        "please provide a valid language"
    )
    LANGUAGE_NOT_SUPPORTED = lambda lang: (
        f"Provider does not support selected language: `{lang}`"
    )


def check_language_format(iso_code):
    """Checks if language code name is formatted correctly (lang-extlang-Script-Reg)"""
    return bool(
        re.fullmatch(
            r"^[a-z]{2,3}(-[a-z]{2,3})?(-[A-Z][a-z]{3})?(-([A-Z]{2,3}|\d{3}))?",
            iso_code,
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


def expand_languages_for_user(list_languages):
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


def load_standardized_language(feature, subfeature, providers: Optional[List[str]] = None):
    """Displays a standardized list of languages for a list of providers
    for the pair (feature, subfeature)"""

    if providers is None:
        interface = import_module('edenai_apis.interface')
        providers = interface.list_providers(feature, subfeature)

    result = []
    for provider in providers:
        list_languages = expand_languages_for_user(
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


