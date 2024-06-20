from typing import Any, Dict
from pydantic import BaseModel
from pydantic_core._pydantic_core import ValidationError
from docstring_parser import parse
from edenai_apis.utils.exception import ProviderException


class OpenAIFunctionTools:
    def __init__(self, tool_name: str, tool_description: str, dataclass: BaseModel):
        self.name = tool_name
        self.description = tool_description
        self.dataclass = dataclass
        self.schema = self.validate_schema()

    def validate_schema(self) -> Dict:
        schema = self.dataclass.model_json_schema()
        docstring = parse(self.dataclass.__doc__ or "")
        parameters = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        for param in docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                if "description" not in parameters["properties"][name]:
                    parameters["properties"][name]["description"] = description

        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )

        return parameters

    def purge_schema(self, field: str, default_value: Any) -> None:
        def recursive_purge(
            schema: Dict[str, Any], field_name: str, default_val: Any
        ) -> None:
            if "properties" in schema:
                for key, value in schema["properties"].items():
                    if key == field_name:
                        schema["properties"][key] = default_val
                    elif "properties" in value or "$ref" in value:
                        recursive_purge(value, field_name, default_val)
            if "$defs" in schema:
                for key, value in schema["$defs"].items():
                    recursive_purge(value, field_name, default_val)

        recursive_purge(self.schema, field, default_value)

    def get_response(self, response: Dict[str, Any]) -> BaseModel:
        try:
            output = self.dataclass.model_validate_json(
                response["choices"][0]["message"]["tool_calls"][0]["function"][
                    "arguments"
                ]
            )
        except ValidationError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return output

    def get_tool(self) -> Dict[str, Any]:
        return [
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.schema,
                },
            }
        ]

    def get_tool_choice(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name}}
