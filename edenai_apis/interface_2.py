from edenai_apis.features import Image as ImageInterface
from edenai_apis.loaders.loaders import load_feature, load_provider
from edenai_apis.loaders.data_loader import FeatureDataEnum, ProviderDataEnum

def return_provider_method(func: callable) -> callable:
    def wrapped(provider:str)-> callable:
        # Get the provider's function. 
        # Example : GoogleAPI
        ProviderClass = load_provider(ProviderDataEnum.CLASS, 
                                       provider_name=provider)

        # Instantiate the provider's class. 
        # Example : google_api = GoogleAPI()
        provider_instance = ProviderClass()
        
        # Get the right function.
        # Example : google_api.image__object_detection
        provider_subfeature_function = getattr(provider_instance, 
                                               func.__name__)

        # TODO : handle exception : not implemented

        return provider_subfeature_function 
    return wrapped

def abstract(InterfaceClass, method_prefix:str):
    class new_abstracted_class:
        pass

    for method_name in dir(InterfaceClass):
        if method_name.startswith(method_prefix):
            # Get the method to overwrite
            attr = getattr(InterfaceClass, method_name)
            # Make it take "provider" as input and return it's method
            wrapped = return_provider_method(attr)
            # Overwriting the method
            setattr(new_abstracted_class, method_name.replace(method_prefix, ""), wrapped)

    return new_abstracted_class

Image = abstract(ImageInterface, method_prefix="image__")
# Text = abstract(ImageInterface, method_prefix="text__")
# Translation = abstract(ImageInterface, method_prefix="translation__")
