"""
    Test language standardization functions.
    Different Providers handle different language formats.
    We implement language utils to handle the standardisation.
"""
from collections import defaultdict
import random
from typing import Sequence
import pytest
import pycountry
from langcodes import Language

from edenai_apis.interface import list_providers
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.tests.test_features_auto import global_method_list_as_params
from edenai_apis.utils.compare import compare
from edenai_apis.utils.languages import (
    get_language_name_from_code,
    load_standardized_language,
    provide_appropriate_language,
    load_language_constraints,
)

def get_languages_iso_code():
    """Returns a list of language codes formed without tags for test"""
    return [
        (language.alpha_2 if hasattr(language, "alpha_2") else language.alpha_3)
        for language in list(pycountry.languages)[:50]
    ]


def get_languages_iso_code_badly_formatted():
    """Returns a sample of language codes badly formated for test"""
    return [
        "en-US-eng",
        "fr-fre-FR-Abcd",
        "FR-fr",
        "ar-arb-arz",
        "en-ennn",
        "ennn",
        "fr-fre-Abcd-FR-",
    ]


def get_languages_iso_code_tags():
    """Returns 50 random made language codes formed with tags for test"""
    languages = list(pycountry.languages)
    scripts = list(pycountry.scripts)
    countries = list(pycountry.countries)
    output = []
    length = min(len(languages), len(scripts), len(countries))
    end_pos = random.randint(100, length)
    start_pos = end_pos - 100
    for i in range(start_pos, end_pos, 2):
        language = (
            languages[i].alpha_2
            if hasattr(languages[i], "alpha_2")
            else languages[i].alpha_3
        )
        # extlang = random.choice(["", "-"+language_1])
        script = random.choice(["", "-" + scripts[i].alpha_4])
        region = countries[i].alpha_2.upper()
        region = random.choice(["", "-" + region])
        output.append(language + script + region)
    return output


class TestLanguageHandling:
    """Tests functions of language handling"""

    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature"),
        global_method_list_as_params()["ungrouped_providers"],
    )
    def test_load_language_constraints(self, provider, feature, subfeature):
        """Tests if all the triplets (provider, feature, subfeature)
        return the correct list of languages constraints"""
        default = defaultdict(lambda: None)
        list_languages = load_language_constraints(provider, feature, subfeature)
        infos = load_provider(ProviderDataEnum.PROVIDER_INFO, provider, feature, subfeature)
        languages = infos.get("constraints", default).get("languages", [])
        if infos.get("constraints", default).get("allow_null_language"):
            languages.append("auto-detect")
        assert compare(
            list_languages,
            languages,
        )

    @pytest.mark.parametrize(
        "iso_language_wrong_format", get_languages_iso_code_badly_formatted()
    )
    def test_provide_appropriate_language_with_wrong_format_iso(
        self, iso_language_wrong_format
    ):
        """Tests if an exception is raised when getting the appropriate language
        from provider languages constraints with an iso_code
        language badly formatted"""
        with pytest.raises(SyntaxError) as exception:
            provide_appropriate_language(
                iso_language_wrong_format,
                provider_name=pytest.PROVIDER,
                feature=pytest.FEATURE,
                subfeature=pytest.SUBFEATURE,
            )
        assert "badly formatted" in str(exception.value)

    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature"),
        global_method_list_as_params()["ungrouped_providers"],
    )
    def test_provider_appropriate_language_with_isocode_none_tags(
        self, provider, feature, subfeature
    ):
        """Tests getting the appropriate language from provider languages
        constraints with an iso_code that is not formed with tags"""
        for iso_language in get_languages_iso_code():
            list_languages: Sequence[str] = load_language_constraints(
                provider, feature, subfeature
            )
            language_selected = provide_appropriate_language(
                iso_language,
                provider_name=provider,
                feature=feature,
                subfeature=subfeature,
            )
            assert language_selected is None or language_selected in list_languages

    @pytest.mark.parametrize(
        ("provider", "feature", "subfeature"),
        global_method_list_as_params()["ungrouped_providers"],
    )
    # @pytest.mark.parametrize('iso_language_tag', get_languages_iso_code_tags())
    def test_provider_appropriate_language_with_isocode_tags(
        self, provider, feature, subfeature
    ):
        """Tests getting the appropriate language from provider
        languages constraints with an iso_code that is formed with tags"""
        language_codes = get_languages_iso_code_tags()
        for iso_language_tag in language_codes:
            language_selected = provide_appropriate_language(
                iso_language_tag,
                provider_name=provider,
                feature=feature,
                subfeature=subfeature,
            )
            assert (
                language_selected is None
                or language_selected == language_codes
                or language_selected
                in load_language_constraints(provider, feature, subfeature)
            )

    @pytest.mark.parametrize(
        ("_provider_i", "feature_i", "subfeature_i"),
        global_method_list_as_params()["ungrouped_providers"],
    )
    def test_load_standardized_language(self, _provider_i, feature_i, subfeature_i):
        """Tests loading standardized languages for user documentation"""
        standardized_language = load_standardized_language(feature_i, subfeature_i)
        providers = list_providers(feature_i, subfeature_i)
        list_languages = []
        for provider in providers:
            list_languages = list(
                set(
                    list_languages
                    + load_language_constraints(provider, feature_i, subfeature_i)
                )
            )
        added_formatted_code_languages = list(
            set(standardized_language) - set(list_languages)
        )
        for language in standardized_language:
            if len(language) != 2:
                if (
                    "-" in language
                    and "Unknown language"
                    not in Language.get(language.split("-")[0]).display_name()
                ):
                    language = language.split("-")[0]
                if len(language) == 3:
                    pycountry_language = pycountry.languages.get(alpha_3=language)
                    if not pycountry_language:
                        language = str(Language.get(language))
                    else:
                        language = (
                            pycountry_language.alpha_2
                            if hasattr(pycountry_language, "alpha_2")
                            else pycountry_language.alpha_3
                        )
                if language not in list_languages:
                    assert language in added_formatted_code_languages

    @pytest.mark.parametrize(
        ("_provider_i", "feature_i", "subfeature_i"),
        global_method_list_as_params()["ungrouped_providers"],
    )
    def test_get_language_name_from_code(self, _provider_i, feature_i, subfeature_i):
        """Tests converting from language code name to a language name"""
        standardized_language = load_standardized_language(feature_i, subfeature_i)
        list_name_language = [
            get_language_name_from_code(language) for language in standardized_language
        ]
        assert len(standardized_language) == len(list_name_language)
        for language_name in list_name_language:
            assert language_name is not None or language_name != ""
