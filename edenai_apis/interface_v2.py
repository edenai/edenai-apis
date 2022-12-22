import inspect
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union, overload
from edenai_apis.features import (
    ImageInterface,
    TextInterface,
    OcrInterface,
    VideoInterface,
    TranslationInterface,
    AudioInterface,
    ProviderInterface as ProviderInterface,
)
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.loaders.data_loader import ProviderDataEnum


def return_provider_method(func: callable) -> callable:
    """TODO Complete this docstring

    Args:
        func (callable): _description_

    Returns:
        callable: _description_
    """
    def wrapped(provider: str)-> callable:
        """TODO Complete this docstring

        Args:
            provider (str): _description_

        Returns:
            callable: _description_
        """
        # Get the provider's function. 
        # Example : GoogleAPI
        ProviderClass = load_provider(ProviderDataEnum.CLASS, provider_name=provider)

        # Instantiate the provider's class. 
        # Example : google_api = GoogleAPI()
        provider_instance = ProviderClass()

        # Get the right function.
        # Example : google_api.image__object_detection
        provider_subfeature_function = getattr(provider_instance, func.__name__)

        return provider_subfeature_function 
    return wrapped

def abstract(InterfaceClass: ProviderInterface, method_prefix: str):
    """TODO Complete this docstring

    Args:
        InterfaceClass (ProviderInterface): _description_
        method_prefix (str): _description_

    Returns:
        _type_: _description_
    """
    class NewAbstractedClass():
        pass

    for method_name in dir(InterfaceClass):
        if method_name.startswith(method_prefix):
            # Get the method to overwrite
            attr = getattr(InterfaceClass, method_name)
            # Make it take "provider" as input and return it's method
            wrapped = return_provider_method(attr)
            # Overwriting the method
            setattr(NewAbstractedClass, method_name.replace(method_prefix, ""), wrapped)

    return NewAbstractedClass

Image = abstract(ImageInterface, method_prefix="image__")
Text = abstract(TextInterface, method_prefix="text__")
Translation = abstract(TranslationInterface, method_prefix="translation__")
Ocr = abstract(OcrInterface, method_prefix="ocr__")
Video = abstract(VideoInterface, method_prefix="video__")
Audio = abstract(AudioInterface, method_prefix="audio__")
