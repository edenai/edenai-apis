"""
EdenAI apis Library

edenai_apis is a library to easily make api calls to various AI providers

Basic usage:
   >>> from edenai_apis import Text
   >>> ner = Text.named_entity_recognition(provider=<provider_name>)
   >>> response = ner(language="en", text="hello world")
You have now access to the provider's response or our own standardize response
   >>> response.standardize_response
    NameEntityRecognitionDataClass(...)
   >>> payload = dict(key1='value1', key2='value2')
   >>> r = requests.post('https://httpbin.org/post', data=payload)
   >>> print(r.text)

To see all available providers and features,
check the Full documentation at <https://edenai.github.io/edenai_apis>.
"""

from . import apis
from . import features
from . import interface
from . import loaders
from . import settings
from . import utils
from .interface_v2 import Text, Ocr, Video, Audio, Image, Translation, Multimodal, Llm
