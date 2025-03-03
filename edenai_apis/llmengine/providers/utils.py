import os
import json
import logging

logging = logging.getLogger(__name__)

PROVIDERS_SETTINGS_BASE_DIR = os.path.join(
    os.path.dirname(__file__), "../..", "providers_keys"
)


def open_settings_file(provider_name: str = None):
    """Open the settings file for the specified provider.

    Args:
        provider_name (str, optional): The name of the provider. Defaults to None.
    """
    if provider_name is None:
        raise ValueError("Provider name is required.")
    provider_name = provider_name.lower()
    try:
        with open(
            os.path.join(PROVIDERS_SETTINGS_BASE_DIR, f"{provider_name}_settings.json")
        ) as settings_file:
            settings = json.load(settings_file)
        return settings
    except FileNotFoundError:
        logging.error(f"Settings file for {provider_name} not found.")
        raise FileNotFoundError(f"Settings file for {provider_name} not found.")
