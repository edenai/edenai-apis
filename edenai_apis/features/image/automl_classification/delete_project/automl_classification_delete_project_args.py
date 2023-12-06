from typing import Dict


def automl_classification_delete_project_arguments(provider_name: str) -> Dict:
    if provider_name == "nyckel":
        project_id = "function_d4eo3fdbjr4u4td4"
    else:
        raise NotImplementedError(
            f"Please add a project id for test arguments of provider: {provider_name}"
        )
    return {"project_id": project_id}
