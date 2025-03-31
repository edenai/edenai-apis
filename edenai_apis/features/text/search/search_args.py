# pylint: disable=locally-disabled, line-too-long
search_sample = {
    "texts": [
        "In Roman mythology, Romulus and Remus (Latin: [ˈroːmʊlʊs], [ˈrɛmʊs]) are twin brothers whose story tells of the events that led to the founding of the city of Rome and the Roman Kingdom by Romulus.",
        "In ancient Roman religion and myth, Mars (Latin: Mārs, pronounced [maːrs]) was the god of war and also an agricultural guardian, a combination characteristic of early Rome.",
        "Proto-Indo-European mythology is the body of myths and deities associated with the Proto-Indo-Europeans, the hypothetical speakers of the reconstructed Proto-Indo-European language",
        "The Ashvamedha (Sanskrit: अश्वमेध, romanized: aśvamedha) was a horse sacrifice ritual followed by the Śrauta tradition of Vedic religion.",
        "Purusha (puruṣa or Sanskrit: पुरुष) is a complex concept whose meaning evolved in Vedic and Upanishadic times. Depending on source and historical timeline, it means the cosmic being or self, consciousness, and universal principle.",
    ],
    "query": "Rome",
}


def search_arguments(provider_name: str):
    return {
        "texts": search_sample["texts"],
        "query": search_sample["query"],
        "similarity_metric": "cosine",
        "model": None,
    }
