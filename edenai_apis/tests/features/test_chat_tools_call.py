import json
import os
import pytest
from edenai_apis import Text


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Skip in opensource package cicd workflow",
)
@pytest.mark.parametrize(
    ("provider", "model"),
    [("mistral", "large-latest"), ("openai", "gpt-4"), ("cohere", "command-nightly")],
)
def test_tool_call(provider, model):
    chat = Text.chat(provider)

    tools = [
        {
            "name": "get_db_result",
            "description": "query a user from db",
            "parameters": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "which user to query",
                    },
                    "is_admin": {
                        "type": "boolean",
                        "description": "query admin user or not",
                    },
                },
                "required": ["location", "is_admin"],
            },
        },
        {
            "name": "get_weather",
            "description": "Get The weather data from a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "description": "The geographical location for which weather information is requested.",
                        "type": "string",
                    },
                    "units": {
                        "description": "The units of measurement for the temperature.",
                        "type": "string",
                        "enum": ["Celsius", "Fahrenheit"],
                        "default": "Celsius",
                    },
                },
                "required": ["location"],
            },
        },
    ]

    tool_call_result = chat(
        text="What is the weather in Brest?",
        chatbot_global_action=None,
        previous_history=None,
        temperature=1,
        max_tokens=2500,
        model=model,
        available_tools=tools,
    )

    tool_call = tool_call_result.standardized_response.message[1].tool_calls[0]
    assert "location" in json.loads(tool_call.arguments)
    assert tool_call.name == "get_weather"

    messages = tool_call_result.standardized_response.message
    history = [msg.model_dump() for msg in messages]

    weather_api_response = {"temperature": "10", "unit": "Celsius", "weather": "rainy"}

    tool_results = [{"id": tool_call.id, "result": weather_api_response}]

    result = chat(
        text="",
        chatbot_global_action=None,
        previous_history=history,
        temperature=1,
        max_tokens=1000,
        model=model,
        tool_results=tool_results,
    )

    result_message = result.standardized_response.generated_text.lower()
    assert "brest" in result_message
    assert "rain" in result_message
