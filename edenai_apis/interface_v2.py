"""
This module serve as an interface to easily make calls to Provider's apis

Example:
    Let's say that we implemented a new Feature Category (3d Models for example)
    and that a few providers already supports it, now we to create an interface for this feature:
    >>> 3DModels = abstract(3dModelsInterface, method_prefix="3dmodeling__")
    this should return a class that can be used as interface and
    we can now use `3DModels` to make call to supported providers
    >>> 3d_from_img = 3DModels.create_3d_model_from_image('<provider_here>')
    >>> response = 3d_from_img(image=...)
"""

from typing import Callable, Dict, Type

from edenai_apis.features import (
    AudioInterface,
    ImageInterface,
    OcrInterface,
    MultimodalInterface,
    LlmInterface,
)
from edenai_apis.features import ProviderInterface
from edenai_apis.features import TextInterface, TranslationInterface, VideoInterface
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider


def return_provider_method(func: Callable) -> Callable:
    """

    Args:
        func (Callable): a ProviderApi method

    Returns:
        Callable: function take a provider_name and return its class's methods
    """

    def wrapped(provider: str, api_keys: Dict = {}) -> Callable:
        """find given func in given provider's class, and returns it

        Args:
            provider (str): provider name

        Returns:
            Callable: provider's function
        """
        # Get the provider's class.
        # Example : GoogleAPI
        ProviderClass = load_provider(ProviderDataEnum.CLASS, provider_name=provider)

        # Instantiate the provider's class.
        # Example : google_api = GoogleAPI()
        provider_instance = ProviderClass(api_keys)

        # Get the right function.
        # Example : google_api.image__object_detection
        provider_subfeature_function = getattr(provider_instance, func.__name__)

        return provider_subfeature_function

    return wrapped


def abstract(InterfaceClass: Type[ProviderInterface], method_prefix: str):
    """create an Abstracted Class and set all the methods of given InterfaceClass
    to it with modified names, methods have the same names as the subfeature

    Args:
        InterfaceClass (ProviderInterface): Provider interface class (that includes abstract methods for each subfeatures)
        method_prefix (str): bit of string to remove from ProvierInterface methods

    Returns:
        NewAbstractedClass: new class with all the renamed methods, that returns `return_provider_method`
    """

    class NewAbstractedClass:
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
Multimodal = abstract(MultimodalInterface, method_prefix="multimodal__")
Llm = abstract(LlmInterface, method_prefix="llm__")
