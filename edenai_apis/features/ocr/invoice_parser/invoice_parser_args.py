import os
from typing import Dict

feature_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(feature_path, "data")


def invoice_parser_arguments() -> Dict:
    return {"file": open(f"{data_path}/invoice.png", "rb"), "language": "en"}
