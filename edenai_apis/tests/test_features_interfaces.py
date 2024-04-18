"""
Test that the interfaces are correctly implemented
"""

import pytest
from edenai_apis.loaders.data_loader import load_class
from inspect import signature


def get_provider_methods():
    providers_classes = load_class()

    for ProviderClass in providers_classes:
        interfaces = ProviderClass.__bases__

        for Interface in interfaces:
            for method_name in getattr(Interface, "__abstractmethods__", []):
                yield (ProviderClass, Interface, method_name)


@pytest.mark.parametrize(
    ("ProviderClass", "Interface", "method_name"), get_provider_methods()
)
def test_inteface_methods_signature(ProviderClass, Interface, method_name):
    """
    Test that implemented provider feature method signature is
    the same as the one in the featur interface
    """
    implemented_method = getattr(ProviderClass, method_name, None)
    interface_method = getattr(Interface, method_name)

    # it is not required to implement all methods of interfaces
    if implemented_method:
        assert signature(implemented_method) == signature(
            interface_method
        ), f"{ProviderClass.__name__}.{method_name} got wront signature, it should match its interface's signature"
