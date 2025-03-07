import importlib, inspect
import os

import logging
from fileinput import filename

logger = logging.getLogger(__name__)

# Get the providers in the providers (this) directory
structure = next(os.walk(os.path.dirname(__file__)))

AVOID_FILES = ["__init__.py", "__pycache__", "pyc", "json"]

LLM_COMPLETION_CLIENTS = {}


logger.info("Loading clients...")


def _extract_client_classes(provider_classes):
    "Takes the classes in the providers and verifies that they're valid"
    for name, cls in provider_classes:
        try:
            client_name = cls.CLIENT_NAME
            if client_name not in LLM_COMPLETION_CLIENTS and client_name != "ignore":
                LLM_COMPLETION_CLIENTS[client_name] = cls
                logger.info(f"Client {client_name} loaded")
        except AttributeError:
            logger.warning(
                f"Client {name} does not have a client name. Are you sure is a valid client?"
            )
            continue


def file_to_avoid(filename: str) -> bool:
    return any([filename.endswith(ext) for ext in AVOID_FILES])


def _find_clients_from_files():
    "Walks the clients package and builds a list of clients classes"
    _classes = []
    for client in structure[1]:
        provider_folder = f"{structure[0]}/{client}"
        provider_file = [
            file
            for file in list(next(os.walk(provider_folder))[2])
            if not file_to_avoid(file)
        ]
        if len(provider_file) > 0:
            # Get the first...
            provider_file = provider_file[0]
            module_name = f"llmengine.clients.{client}.{provider_file[:-3]}"
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
                logger.warning(
                    f"Module {module_name} has a problem an cannot be instantiated"
                )
            except ModuleNotFoundError as e:
                logger.warning(f"Module {module_name} not found: {e}")
    return _classes


_extract_client_classes(_find_clients_from_files())
