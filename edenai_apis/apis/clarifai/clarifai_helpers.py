def get_formatted_language(language: str):
    return {
        "ru-RU": "ocr-scene-cyrillic-paddleocr",
        "pt-PT": "ocr-scene-portuguese-paddleocr",
        "ko-KR": "ocr-scene-korean-paddleocr",
        "ja-JP": "ocr-scene-japanese-paddleocr",
        "it-IT": "ocr-scene-italian-paddleocr",
        "fr-FR": "ocr-scene-french-paddleocr",
        "de-DE": "ocr-scene-german-paddleocr",
        "zh-TW": "ocr-scene-chinese-traditional-paddleocr",
        "zh-CN": "ocr-scene-chinese-english-paddleocr",
        "ar-XA": "ocr-scene-arabic-paddleocr",
        "es-ES": "ocr-scene-spanish-paddleocr",
        "en-US": "ocr-scene-english-paddleocr",
    }[language]


def explicit_content_likelihood(value):
    value = value * 100
    if value < 10:
        return 1
    elif value < 30:
        return 2
    elif value < 60:
        return 3
    elif value < 80:
        return 4
    elif value > 80:
        return 5
    else:
        return 0
