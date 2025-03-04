from llmengine.clients.llm_models.model_management_interface import (
    ModelManagementInterface,
)
from litellm import get_model_info


class TestModelManagementInterface:

    def test_is_valid_model_info(
        self, unknown_models_to_litellm, invalid_models_to_litellm
    ):
        assert (
            ModelManagementInterface.is_valid_model_info(unknown_models_to_litellm)
            is not None
        )
        assert (
            ModelManagementInterface.is_valid_model_info(invalid_models_to_litellm)
            is None
        )

    def test_try_register_model(self, unknown_models_to_litellm):
        model_management_interface = ModelManagementInterface()
        assert (
            model_management_interface.try_register_model(unknown_models_to_litellm)
            is True
        )
        # assert model_management_interface.try_register_model("test_model", "test_model") is False

    def test_try_register_invalid_model(self, invalid_models_to_litellm):
        model_management_interface = ModelManagementInterface()
        assert (
            model_management_interface.try_register_model(invalid_models_to_litellm)
            is False
        )

    def test_update_registered_model(
        self, unknown_models_to_litellm, update_known_models_to_litellm
    ):
        model_management_interface = ModelManagementInterface()
        assert (
            model_management_interface.try_register_model(unknown_models_to_litellm)
            is True
        )
        assert (
            model_management_interface.update_registered_model(
                update_known_models_to_litellm
            )
            is True
        )
        info = get_model_info(model="test-inexisting-model")
        for model_info in update_known_models_to_litellm.values():
            for key, value in model_info.items():
                assert info[key] == value
