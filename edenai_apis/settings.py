import os
import sys

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

base_path = os.path.dirname(__file__)
sys.path.append(base_path)

tests_path = os.path.join(base_path, "tests")
keys_path = os.path.join(base_path, os.environ.get("API_KEYS_PATH", "provider_settings"))
template_keys_path = os.path.join(base_path, "provider_settings")
apis_path = os.path.join(base_path, "apis")
features_path = os.path.join(base_path, "features")
outputs_path = lambda provider: os.path.join(apis_path, provider, "outputs")
info_path = lambda provider: os.path.join(apis_path, provider, "info.json")
loader_path = os.path.join(base_path, "loaders_new", "data_loader")
