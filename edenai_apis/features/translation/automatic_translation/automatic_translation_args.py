def automatic_translation_arguments(provider_name: str):
    return {
        "text": "人工智能 亦稱智械、機器智能，指由人製造出來的機器所表現出來的智慧。通常人工智能是指通过普通電腦程式來呈現人類智能的技術。"
        + "該詞也指出研究這樣的智能系統是否能夠實現，以及如何實現。同时，通過醫學、神經科學、機器人學及統計學等的進步，常態預測則認為人類的很多職業也逐漸被其取代",
        "source_language": "zh",
        "target_language": "en",
        "settings": {"openai": "gpt-4o", "xai": "grok-2-latest"},
    }
