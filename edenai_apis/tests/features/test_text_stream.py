import os
from typing import Iterator

import pytest

from edenai_apis import Text
from edenai_apis.features.text.chat import StreamChat, ChatStreamResponse
from edenai_apis.interface import list_providers
from edenai_apis.loaders.data_loader import FeatureDataEnum
from edenai_apis.loaders.loaders import load_feature
from edenai_apis.utils.constraints import validate_all_provider_constraints

chat_providers = sorted(list_providers(feature="text", subfeature="chat"))

chat_provider_models = [
    ("google", "chat-bison"),
    ("openai", "gpt-3.5-turbo"),
    ("mistral", "large-latest"),
    ("anthropic", "claude-3-sonnet-20240229-v1:0"),
    ("cohere", "command-nightly"),
    ("meta", "llama2-70b-chat-v1"),
    ("perplexityai", "pplx-70b-online"),
    ("replicate", "llama-2-70b-chat"),
    ("deepseek", "DeepSeek-V3"),
]


def test_provider_models_coverage():
    for provider in chat_providers:
        assert provider in list(map(lambda x: x[0], chat_provider_models))


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Skip in opensource package cicd workflow",
)
@pytest.mark.parametrize(("provider", "model"), chat_provider_models)
class TestImageSearch:
    def test_stream(self, provider, model):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="text",
            subfeature="chat",
            provider_name=provider,
        )
        feature_args["stream"] = True
        feature_args["settings"] = {provider: model}
        feature_args = validate_all_provider_constraints(
            provider, "text", "chat", None, feature_args
        )

        chat = Text.chat(provider)
        chat_output = chat(**feature_args)

        assert isinstance(chat_output.standardized_response, StreamChat)
        assert isinstance(chat_output.standardized_response.stream, Iterator)

        for chunk in chat_output.standardized_response.stream:
            assert isinstance(chunk, ChatStreamResponse)
