import json
import re
from typing import Dict, List, Optional, Tuple, Union

from edenai_apis.features.text.chat.helpers import get_tool_call_from_history_by_id


def extract_json_text(input_string: str) -> Optional[Union[dict, list]]:
    if "[" in input_string and "]" in input_string:
        start_index = input_string.index("[")
        end_index = input_string.rindex("]")
        json_text = input_string[start_index : end_index + 1]
        json_text = json_text.replace("'", '"')
        return json.loads(json_text)
    elif input_string[0] == "{":
        match = re.search(r"\{.*\}", input_string)
        if match:
            json_text = match.group(0)
            return json.loads(json_text)
    else:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, input_string, re.DOTALL)

        if match:
            json_text = match.group(1).strip()
            return json.loads(json_text)
        else:
            return None


json_schema_types_map_python_types = {
    "string": "str",
    "number": "Union[float, int]",
    "integer": "int",
    "boolean": "bool",
    "object": "Dict",
    "array": "List",
}


def _end_of_description_sentence(description):
    return ". " if description and not description.strip().endswith(".") else " "


def get_type(json_schema_param, base_description="") -> Tuple[str, str]:
    """
    Recursively construct type for cohere tool parameters by translating json-schema into python types
    For nested dict or list, we add the parameter description to the original description

    Returns:
        type: str, description: str
    """
    param_type = json_schema_types_map_python_types.get(
        json_schema_param["type"], "Any"
    )
    description = base_description + json_schema_param.get("description", "")

    # ref: https://docs.cohere.com/docs/parameter-types-in-tool-use#example--arrays
    if json_schema_param["type"] == "array":
        if items := json_schema_param["items"]:
            array_param, description = get_type(items, description)
            return f"List[{array_param}]", description

    # ref: https://docs.cohere.com/docs/parameter-types-in-tool-use#example--dictionaries
    if json_schema_param["type"] == "object":
        if properties := json_schema_param["properties"]:
            dict_values_types = set()
            end_of_sentence = _end_of_description_sentence(description)
            description += (
                f"{end_of_sentence}Contains the following list of attributes: \n"
            )

            # get type and add each key, value to the description of this param
            for key, value in properties.items():
                dict_value_type, item_description = get_type(value)
                dict_values_types.add(dict_value_type)
                description += f"- {key}: {item_description} \n,"

            if len(dict_value_type) == 1:
                final_value_type = list(dict_values_types)[0]
            else:
                final_value_type = f"Union[{' ,'.join(dict_values_types)}]"
            return f"Dict[str, {final_value_type}]", description

    # ref: https://docs.cohere.com/docs/parameter-types-in-tool-use#example--enumerated-values-enums
    if enum := json_schema_param.get("enum"):
        end_of_sentence = _end_of_description_sentence(description)
        description += f"{end_of_sentence}Possible enum values: {', '.join(enum)}."

    # ref: https://docs.cohere.com/docs/parameter-types-in-tool-use#example---defaults
    if (default_val := json_schema_param.get("default")) is not None:
        end_of_sentence = _end_of_description_sentence(description)
        description += f"{end_of_sentence}The default value is: {default_val}."

    return param_type, description


def convert_tools_to_cohere(available_tools):
    tools = []
    for tool in available_tools:
        parameters_definitions = {}
        for name, prop in tool["parameters"]["properties"].items():
            param = {}
            param_type, description = get_type(prop)
            param["type"] = param_type
            param["description"] = description
            param["required"] = name in tool["parameters"]["required"]
            parameters_definitions[name] = param

        tools.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameter_definitions": parameters_definitions,
            }
        )
    return tools


def convert_cohere_tool_call_to_edenai_tool_call(tool_call):
    return {
        "name": tool_call["name"],
        "parameters": json.loads(tool_call["arguments"]),
        "generation_id": "-".join(tool_call["id"].split("-")[:-2]),
    }


def convert_tools_results_to_cohere(
    tools_results: List[Dict[str, str]], previous_history
):
    if not tools_results:
        return None

    result = []
    for tool_result in tools_results:
        tool_call = get_tool_call_from_history_by_id(
            tool_result["id"], previous_history
        )
        tool_output = tool_result["result"]
        call = convert_cohere_tool_call_to_edenai_tool_call(tool_call)
        output = [{"result": tool_output}]
        result.append({"call": call, "outputs": output})
    return result


cohere_roles = {
    "user": "USER",
    "system": "SYSTEM",
    "assistant": "CHATBOT",
    "chatbot": "CHATBOT",
    "tool": "TOOL",
}
