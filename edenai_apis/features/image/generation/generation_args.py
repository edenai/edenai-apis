from typing import Dict


def generation_arguments(provider_name: str) -> Dict:
    return {
        "text": "A huge red ballon flying outside the city.",
        "resolution": "1024x1024",
        "num_images": 1,
    }
