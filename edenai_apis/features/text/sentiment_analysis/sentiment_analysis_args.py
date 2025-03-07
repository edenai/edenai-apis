# pylint: disable=locally-disabled, line-too-long
def sentiment_analysis_arguments(provider_name: str):
    return {
        "language": "en",
        "text": "Overall I am satisfied with my experience at Amazon, but two areas of major improvement needed. First is the product reviews and pricing. There are thousands of positive reviews for so many items, and it's clear that the reviews are bogus or not really associated with that product. There needs to be a way to only view products sold by Amazon directly, because many market sellers way overprice items that can be purchased cheaper elsewhere (like Walmart, Target, etc). The second issue is they make it too difficult to get help when there's an issue with an order.",
        "settings": {"openai": "gpt-4o", "xai": "grok-2"},
    }
