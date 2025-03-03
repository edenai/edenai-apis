from openai._models import BaseModel as OpenAI_BaseModel
from typing import Dict, Optional


class PropertyType(OpenAI_BaseModel):
    def __init__(
        self, type: str, description: str, enum: Optional[list] = None, **params
    ):
        super(PropertyType, self).__init__(**params)
        self.type = type
        self.description = description
        if enum is not None:
            self.enum = enum


class FunctionParameterType(OpenAI_BaseModel):
    def __init__(
        self,
        type: str,
        properties: Dict[str, PropertyType],
        required: Optional[list] = [],
        **params,
    ):
        super(FunctionParameterType, self).__init__(**params)
        self.type = type
        self.properties = properties
        self.required = required


class FunctionType(OpenAI_BaseModel):
    def __init__(
        self, name: str, description: str, parameters: FunctionParameterType, **params
    ):
        super(FunctionType, self).__init__(**params)
        self.name = name
        self.description = description
        self.parameters = parameters


class ToolType(OpenAI_BaseModel):
    def __init__(self, type: str, function: FunctionType, **params):
        super(ToolType, self).__init__(**params)
        self.type = type
        self.function = function

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)
