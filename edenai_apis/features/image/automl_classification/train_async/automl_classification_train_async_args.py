from typing import Dict


def automl_classification_train_async_arguments(provider_name: str) -> Dict:
    if provider_name == "nyckel":
        project_id = "function_dlsxq3ze6ht4480n"
    else:
        raise NotImplementedError(
            f"Please add a project id for test arguments of provider: {provider_name}"
        )
    return {"project_id": project_id}
