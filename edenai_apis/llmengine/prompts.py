import json
import aiofiles
from pydantic import BaseModel
from settings import base_path


class BasePrompt:

    @staticmethod
    def compose_prompt(
        behavior: str, example_file: str, dataclass: BaseModel, **kwargs
    ):

        with open(f"{base_path}/features/{example_file}", "r") as f:
            output_response = json.load(f)

        system_messages = [
            {
                "role": "system",
                "content": f"{behavior}. You return a json contructuted with double-quotes. Double quotes within strings must be escaped with backslash, single quotes within strings will not be escaped. You must provide a complete parseable JSON. The output shaped like the following with the exact same structure and the exact same keys but the values would change: \n {output_response}",
            }
        ]
        if dataclass is not None:
            system_messages[0]["content"] = (
                "{} \n\n You must follow this pydantic dataclass schema {}".format(
                    system_messages[0]["content"], dataclass.model_json_schema()
                )
            )
        return system_messages

    @staticmethod
    async def acompose_prompt(
        behavior: str, example_file: str, dataclass: BaseModel, **kwargs
    ):

        async def load_json_async(file_path):
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
            return json.loads(content)

        output_response = await load_json_async(f"{base_path}/features/{example_file}")

        system_messages = [
            {
                "role": "system",
                "content": f"{behavior}. You return a json contructuted with double-quotes. Double quotes within strings must be escaped with backslash, single quotes within strings will not be escaped. You must provide a complete parseable JSON. The output shaped like the following with the exact same structure and the exact same keys but the values would change: \n {output_response}",
            }
        ]
        if dataclass is not None:
            system_messages[0]["content"] = (
                "{} \n\n You must follow this pydantic dataclass schema {}".format(
                    system_messages[0]["content"], dataclass.model_json_schema()
                )
            )
        return system_messages
