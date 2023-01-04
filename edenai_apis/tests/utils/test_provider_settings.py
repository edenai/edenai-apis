import pytest
import os
from edenai_apis.interface import list_providers
from edenai_apis.settings import template_keys_path


@pytest.mark.parametrize("provider", list_providers())
def test_no_missing_provider_settings_template(provider):
    """Assert that template setting file exists for given provider"""
    assert f"{provider}_settings.json" in os.listdir(
        template_keys_path
    ), (
        f"Provider {provider} is missing a settings template file."
        f"Please add it under {template_keys_path}/{provider}_settings.json, "
        f"or launch the script to regenerate all settings templates: "
        "`python -m edenai_apis.utils.api_keys_templates`"
    )
