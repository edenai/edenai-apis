import logging
from typing import Union
from litellm import register_model, get_model_info

logger = logging.getLogger(__name__)


class ModelManagementInterface:

    CLIENT_NAME = "ignore"

    def __init__(self):
        pass

    @staticmethod
    def is_valid_model_info(model_info: dict) -> Union[str, None]:
        if model_info is None:
            return None
        if not isinstance(model_info, dict):
            return None
        model_name = list(model_info.keys())[0]
        if not isinstance(model_info[model_name], dict):
            return None
        return model_name

    def try_register_model(self, model_info: dict):
        model_name = ModelManagementInterface.is_valid_model_info(model_info=model_info)
        if model_name is None:
            return False
        try:
            existing_model_data = get_model_info(model_name)
            logger.info(f"Model: {model_name} already registered")
            return True
        except Exception as ex:
            logger.warning(f"The model: {model_name} hasn't been registered: {ex}")
            existing_model_data = None

        if existing_model_data is None:
            try:
                register_model(model_info)
                logger.info(f"Registered the model: {model_name}")
                return True
            except Exception as ex:
                logger.error(f"Failed to register the model: {model_name}: {ex}")
                return False

    def update_registered_model(self, model_info: dict):
        model_name = ModelManagementInterface.is_valid_model_info(model_info=model_info)
        if model_name is None:
            return False
        try:
            existing_model_data = get_model_info(model=model_name)
            new_model = {}
            new_model[model_name] = {}
            for key, value in model_info[model_name].items():
                if key == "key":
                    continue
                new_model[key] = existing_model_data[key]
                if key in new_model:
                    new_model[model_name][key] = value
            for model_info_ in model_info.values():
                for key, value in model_info_.items():
                    new_model[model_name][key] = value
            register_model(new_model)
            return True
        except Exception as ex:
            logger.warning(f"The model: {model_name} hasn't been registered: {ex}")
            return self.try_register_model(model_info=model_info)
