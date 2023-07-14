#!/usr/bin/env python3
"""
Generate feature/subfeature list as well as list of all available providers
to be added to the README.md file
"""
import os
from typing import List
from edenai_apis.interface import ProviderDict, ProviderList, list_features
from settings import base_path


def format_feature_list_to_feature_dict(feature_list: ProviderList):
    """convert list of features to a markdown table
    Listing features, then subfeature as subcategory
    then list of providers available for these subfeature

    Returns Mardown formatted string
    """
    feature_dict = {}
    for provider, feature, subfeature, *phase in feature_list:
        feature_dict[feature] = feature_dict.get(feature, {})
        feature_dict[feature][subfeature] = feature_dict[feature].get(subfeature, {})
        feature_dict[feature][subfeature][provider] = True
    return feature_dict


def dict_to_markdown_table(
    provider_feature_dict: ProviderDict, first_header, second_header
) -> str:
    """convert dict of providers or features to a markdown table

    Listing providers, then features as subcategory
    then list of subfeatures available for these features
    or
    Listing features, then subfeature as subcategory
    then list of providers available for these subfeature

    Returns Mardown formatted string
    """
    text = ""
    for header, categories in provider_feature_dict.items():
        table = f"\n| {first_header} | {second_header} |\n|----------|-------------|\n"
        for category, subcategories in categories.items():
            for index, subcategory in enumerate(subcategories.keys()):
                if index == 0:
                    table += f"| **{category}** | {subcategory} |\n"
                else:
                    table += f"| | {subcategory} |\n"
        text += f"<details><summary>{header}</summary>\n" f"{table}\n" "</details>\n"
    return text


def main():
    """write outputs to a markdown file"""
    MARKDOWN_FILE_NAME = "AVAILABLES_FEATURES_AND_PROVIDERS.md"
    path = os.path.join(base_path, "..", MARKDOWN_FILE_NAME)
    print(f"=== Generating {MARKDOWN_FILE_NAME} ===")
    with open(path, "w") as doc:
        doc.write("")  # emtpy file

    with open(path, "a+") as doc:
        feature_dict = format_feature_list_to_feature_dict(list_features())
        doc.write("# Available Features:\n")
        doc.write(dict_to_markdown_table(feature_dict, "Subfeatures", "Providers"))

        doc.write("\n\n")

        provider_dict = list_features(as_dict=True)
        doc.write("# Available Providers:\n")
        doc.write(dict_to_markdown_table(provider_dict, "Features", "Subfeatures"))


if __name__ == "__main__":
    main()

provider_list = list_features(as_dict=True)
