import pytest

import json
import pathlib
import responses

from edenai_apis.apis.corticalio.client import CorticalClient
from edenai_apis.apis.corticalio.helpers import normalize_keywords
from edenai_apis.utils.exception import ProviderException


@pytest.mark.corticalio
class TestCorticalClient:

    @pytest.fixture
    def cortical_client(self):
        return CorticalClient({"api_key": "abc", "base_url": "http://api.cortical.io/"})

    @pytest.fixture
    def keywords_cortical(self):
        api_folder = pathlib.Path(__file__).parent.parent.resolve()
        outputs_path = (
            api_folder / "outputs" / "text" / "keyword_extraction_output.json"
        )
        with open(outputs_path) as keywords_file:
            return json.load(keywords_file)

    def test_keyword_normalization(self, keywords_cortical):
        original_response = keywords_cortical.get("original_response")
        normalised_response = normalize_keywords(original_response)
        standardized_response = keywords_cortical.get("standardized_response")

        assert "items" in standardized_response

        normalised_items = normalised_response.items
        standardized_items = standardized_response.get("items")

        assert len(normalised_items) == len(standardized_items)
        for n_item in normalised_items:
            s_item = [
                i for i in standardized_items if n_item.keyword == i.get("keyword")
            ][0]
            assert n_item.keyword == s_item.get("keyword")
            assert n_item.importance == s_item.get("importance")
            assert len(str(n_item.importance)) <= 4

    @responses.activate
    def test_failed_request(self, cortical_client):
        responses.add(responses.POST, "http://api.cortical.io/keywords", status=400)

        with pytest.raises(ProviderException) as e:
            cortical_client.extract_keywords("", "en")

    @responses.activate
    def test_text_sample(self, cortical_client, keywords_cortical):
        keywords_response = keywords_cortical.get("original_response")
        responses.add(
            responses.POST,
            "http://api.cortical.io/keywords",
            status=200,
            json=keywords_response,
        )

        text_folder = pathlib.Path(__file__).parent.resolve()
        sample_path = text_folder / "resources" / "text_sample.txt"
        with open(sample_path) as text_file:
            text_sample = text_file.read()
            assert cortical_client.extract_keywords(text_sample, "en")
