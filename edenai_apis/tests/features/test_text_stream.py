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


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Skip in opensource package cicd workflow",
)
@pytest.mark.parametrize(("provider"), chat_providers)
class TestImageSearch:
    def test_stream(self, provider):
        feature_args = load_feature(
            FeatureDataEnum.SAMPLES_ARGS,
            feature="text",
            subfeature="chat",
            provider_name=provider
        )
        feature_args['stream'] = True
        feature_args = validate_all_provider_constraints(
            provider, "text", "chat", None, feature_args
        )

        chat = Text.chat(provider)
        chat_output = chat(**feature_args)

        assert isinstance(chat_output.standardized_response, StreamChat)
        assert isinstance(chat_output.standardized_response.stream, Iterator)

        for chunk in chat_output.standardized_response.stream:
            assert isinstance(chunk, ChatStreamResponse)
