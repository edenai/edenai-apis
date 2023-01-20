import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def custom_document_parsing_arguments() -> Dict:
    return {"file": open(f"{data_path}/resume.pdf", "rb"),
            "queries": [
                {
                    "query" : "What is the resume's email address?",
                    "pages" : "1-*"
                },
                {
                    "query" : "What is the first Adult Care experience?",
                    "pages" : "1"
                }
            ]
        }
