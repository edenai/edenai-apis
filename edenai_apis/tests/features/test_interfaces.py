"""
Test that the interfaces are correctly implemented
"""

from inspect import signature

import pytest

from edenai_apis.loaders.data_loader import load_class

import difflib


def get_provider_methods():
    providers_classes = load_class()

    for ProviderClass in providers_classes:
        interfaces = ProviderClass.__bases__
        for Interface in interfaces:
            for method_name, method in Interface.__dict__.items():
                if getattr(method, "__isabstractmethod__", False):
                    yield (ProviderClass, Interface, method_name)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("ProviderClass", "Interface", "method_name"),
    sorted(get_provider_methods(), key=lambda param: param[2]),
)
def test_inteface_methods_signature(ProviderClass, Interface, method_name):
    """
    Test that implemented provider feature method signature is
    the same as the one in the feature interface
    """
    implemented_method = getattr(ProviderClass, method_name, None)
    interface_method = getattr(Interface, method_name)

    # it is not required to implement all methods of interfaces
    if implemented_method and method_name != "text__chat":
        assert signature(implemented_method) == signature(
            interface_method
        ), f"{ProviderClass.__name__}.{method_name} got wrong signature, it should match its interface's signature"
