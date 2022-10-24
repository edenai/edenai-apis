def get_formatted_language(language: str):
    return {
        "ru-RU": "82c45a5c3b4468ed09903a2b364d60f0",
        "pt-PT": "731d5af335b9e11642aee5ce1df9e66f",
        "ko-KR": "7e911b5660214c4b8e7bbc67e473a9cb",
        "ja-JP": "10da02d313f8a42ad619f8acbe6501fc",
        "it-IT": "ebf17633e16b3c3c97f4b7f80467875d",
        "fr-FR": "0af97f8746e366611e579dffaee65a2d",
        "de-DE": "7fd43ae450cb29e0b9f51aa5c44801d9",
        "zh-TW": "9462d96a382f8c64d4eb4e2ea5cfe455",
        "zh-CN": "ee46555bf1257d6653e21b1e61024ed2",
        "ar-XA": "f4cea61053fe8abff5ae88d8b3200c23",
        "es-ES": "193ef6ea59761b6d4306d9adc06e96a1",
        "en-US": "f1b1005c8feaa8d3f34d35f224092915",
    }[language]


def explicit_content_likelihood(value):
    value = value * 100
    if value < 10:
        return 1
    elif value < 30:
        return 2
    elif value < 60:
        return 3
    elif  value < 80:
        return 4
    elif value > 80:
        return 5
    else:
        return 0
