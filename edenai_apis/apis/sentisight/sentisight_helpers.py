from typing import Dict, Sequence


def get_formatted_language(language: str):
    if language in ["sr", "zh-CN", "zh-TW"]:
        return {
            "sr": "rs_latin",
            "zh-CN": "ch_sim",
            "zh-TW": "ch_tra",
        }[language]
    return language


def calculate_bounding_box(points: Sequence[Dict], max_width, max_height):
    return {
        "x": min(points, key=lambda x: x["x"])["x"] / max_width,
        "y": max(points, key=lambda x: x["y"])["y"] / max_height,
        "width": max(points, key=lambda x: x["x"])["x"] / max_width
        - min(points, key=lambda x: x["x"])["x"] / max_width,
        "height": max(points, key=lambda x: x["y"])["y"] / max_height
        - min(points, key=lambda x: x["y"])["y"] / max_height,
    }
