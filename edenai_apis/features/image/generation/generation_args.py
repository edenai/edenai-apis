from typing import Dict


def generation_arguments(provider_name: str) -> Dict:
    return {
        "text": "A huge red ballon flying outside the city.",
        "resolution": "1024x1024",
        "num_images": 1,
        "settings": {
            "amazon": "titan-image-generator-v1_premium",
            "openai": "dall-e-3",
            "stabilityai": "stable-diffusion-xl-1024-v1-0",
            "replicate": "classic",
        },
    }
