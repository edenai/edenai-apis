import json
import re
from typing import Optional, Union


def extract_json_text(input_string: str) -> Optional[Union[dict, list]]:
    pattern = r"```json(.*?)```"
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        json_text = match.group(1).strip()
        return json.loads(json_text)
    else:
        return None
