import importlib, inspect
import os

import logging

# Get the providers in the providers (this) directory
structure = next(os.walk(os.path.dirname(__file__)))

AVOID_FILES = ["__init__.py", "__pycache__"]

LLM_PROVIDERS = {}


def _extract_provider_classes(provider_classes):
    "Takes the classes in the providers and verifies that they're valid"
    for name, cls in provider_classes:
        try:
            provider_name = cls.get_provider_name()
            if provider_name not in LLM_PROVIDERS:
                LLM_PROVIDERS[provider_name] = cls
        except AttributeError:
            logging.warning(
                f"Provider {name} does not have a provider name. Are you sure is a valid provider name"
            )
            continue


def _find_providers_from_files():
    "Walks the providers package and builds a list of provider classes"
    _classes = []
    for provider in structure[1]:
        provider_folder = f"{structure[0]}/{provider}"
        provider_file = list(
            filter(
                lambda name: name not in AVOID_FILES, next(os.walk(provider_folder))[2]
            )
        )
        # Get the first...
        provider_file = provider_file[0]
        module_name = f"llm_engine.providers.{provider}.{provider_file[:-3]}"
        try:
            for name, cls in inspect.getmembers(
                importlib.import_module(module_name, inspect.isclass)
            ):
                try:
                    if cls.__module__ == module_name:
                        _classes.append((name, cls))
                except AttributeError:
                    continue
        except NameError:
            logging.warning(
                f"Module {module_name} has a problem an cannot be instantiated"
            )
        except ModuleNotFoundError as e:
            logging.warning(f"Module {module_name} not found: {e}")
    return _classes


# _extract_provider_classes(_find_providers_from_files())
