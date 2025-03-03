# Try to register the new models for litellm
from pathlib import Path
from litellm import register_model
import json
import logging

logger = logging.getLogger(__name__)


LITELLM_MODELS_FILE = f"{Path(__file__).parent.parent.resolve()}/llm_models/models.json"


def register_litellm_models():

    with open(LITELLM_MODELS_FILE, "r") as f:
        models = json.load(f)
        model_names = []
        for model_name, model in models.items():
            cost = {}
            cost[model_name] = model
            try:
                result = register_model(cost)
                model_names.append(cost)
            except Exception as e:
                logger.error(f"Error registering model {model_name}: {e}")
        return model_names


register_litellm_models()
