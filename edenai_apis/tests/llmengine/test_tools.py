import pytest

from llmengine.types.tools_types import (
    ToolType,
    PropertyType,
    FunctionParameterType,
    FunctionType,
)
from llmengine.llm_engine import LLMEngine


def factorial(n):
    if n > 10:
        raise "Cannot do more than 10 in this test case"
    if n < 0:
        raise "Cannot do negative numbers"
    if n == 0:
        return 1
    return n * factorial(n - 1)


@pytest.fixture
def engine():
    return LLMEngine(
        provider_name="openai", model_name="openai/gpt-3.5-turbo", client_name="litellm"
    )


class TestTools:

    def test_tool_type_instantiation(self):
        test_tool = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        }
        properties = {
            "location": PropertyType(
                **test_tool["function"]["parameters"]["properties"]["location"]
            ),
            "format": PropertyType(
                **test_tool["function"]["parameters"]["properties"]["format"]
            ),
        }
        params = FunctionParameterType(
            type=test_tool["function"]["parameters"]["type"],
            properties=properties,
            required=test_tool["function"]["parameters"]["required"],
        )
        function = FunctionType(
            name=test_tool["function"]["name"],
            description=test_tool["function"]["description"],
            parameters=params,
        )
        tool = ToolType(type="function", function=function)
        assert tool.type == "function"
        assert tool.function.name == "get_current_weather"

    def test_tool_type_instantiation(self):
        test_tool = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
        }

        function_to_change = {
            "name": "get_current_lunar_weather",
            "description": "Get the current weather in the moon",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
        properties = {
            "location": PropertyType(
                **test_tool["function"]["parameters"]["properties"]["location"]
            ),
            "format": PropertyType(
                **test_tool["function"]["parameters"]["properties"]["format"]
            ),
        }
        params = FunctionParameterType(
            type=test_tool["function"]["parameters"]["type"],
            properties=properties,
            required=test_tool["function"]["parameters"]["required"],
        )
        function = FunctionType(
            name=test_tool["function"]["name"],
            description=test_tool["function"]["description"],
            parameters=params,
        )
        tool = ToolType(type="function", function=function)
        assert tool.type == "function"
        assert tool.function.name == "get_current_weather"
        # Change the name of the function
        tool.function = FunctionType(**function_to_change)
        assert tool.function.name == "get_current_lunar_weather"

    def test_tool_use(self, engine: LLMEngine):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "factorial",
                    "description": "Calculate the factorial of an integer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "The integer to calculate the factorial of",
                            },
                        },
                        "required": ["n"],
                    },
                },
            }
        ]

        tool_calls = [
            {
                "id": "call_12345xyz",
                "type": "function",
                "function": {"name": "factorial", "arguments": '{"n":"5"}'},
            }
        ]

        params = {
            "chatbot_global_action": "You are a helpful assistant. You should use tools for calculating the factorial of an integer greater that 2",
            "text": "What is 5!",
            "available_tools": tools,
            "tool_choice": "required",
            "previous_history": [],
            "temperature": 0.0,
            "max_tokens": 1000,
            "model": "gpt-3.5-turbo",
            "mock_tool_calls": tool_calls,
            "api_key": "opeizaopei",
        }
        response = engine.chat(**params)
        assert (
            response.original_response["choices"][0]["message"]["tool_calls"][0][
                "function"
            ]["name"]
            == "factorial"
        )
