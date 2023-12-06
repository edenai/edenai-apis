from typing import Dict


def automl_classification_train_async_arguments(provider_name: str) -> Dict:
    if provider_name == "nyckel":
        project_id = "function_yfisrgk70k1iuroq"
    else:
        raise NotImplementedError(
            f"Please add a project id for test arguments of provider: {provider_name}"
        )
    return {"project_id": project_id}
