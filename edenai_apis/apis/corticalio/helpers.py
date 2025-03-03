from typing import Dict

from edenai_apis.features.text import (
    KeywordExtractionDataClass,
    InfosKeywordExtractionDataClass,
)


def normalize_keywords(response: Dict):
    items = [
        InfosKeywordExtractionDataClass(
            keyword=keyword["word"], importance="{:.2f}".format(keyword["score"])
        )
        for keyword in response["keywords"]
    ]

    return KeywordExtractionDataClass(items=items)
