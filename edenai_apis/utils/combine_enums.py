from typing import Type
from enum import Enum


def combine_enums(name: str, *enum_classes: Type[Enum]) -> Enum:
    combined = {}
    for enum_class in enum_classes:
        combined.update(enum_class.__members__)
    return Enum(name, combined)
