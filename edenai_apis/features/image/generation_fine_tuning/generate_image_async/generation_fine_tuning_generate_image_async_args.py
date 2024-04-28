from typing import Dict


def generation_fine_tuning_generate_image_async_arguments(provider_name: str) -> Dict:
    project_id = ""
    if provider_name == "astria":
        project_id = "test"
    else:
        raise NotImplementedError(
            f"Please add a project id for test arguments of provider: {provider_name}"
        )
    return {"project_id": project_id, "prompt": "Smiling cat"}
