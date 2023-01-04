#!/usr/bin/env python3
import json
import os
from typing import Any, Union
from edenai_apis.interface import list_providers
from edenai_apis.settings import keys_path, template_keys_path


def create_templates_from_settings_files():
    """create empty templates from all providers settings files"""
    file_names = [f"{provider}_settings.json" for provider in list_providers()]
    for setting_file_name in file_names:

        with open(os.path.join(keys_path, setting_file_name), "r") as setting_file:
            data: dict = json.load(setting_file)
            blanked_settings = blank_values(data)

            with open(os.path.join(template_keys_path, setting_file_name), "w") as template_file:
                text = json.dumps(blanked_settings, indent=2)
                template_file.write(text)


def blank_values(data: Union[dict, Any]):
    """receive a dict and recursively put blank string in its values"""
    if isinstance(data, dict):
        return {key: blank_values(value) for key, value in data.items()}
    return ""


if __name__ == "__main__":
    create_templates_from_settings_files()
