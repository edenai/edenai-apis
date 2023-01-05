import os

feature_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(feature_path, "data")


def document_translation_arguments() -> dict:
    return {
        "file": open(f'{data_path}/document_translation.pdf', 'rb'),
        "source_language": "en",
        "target_language": "fr"
    }