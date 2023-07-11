from abc import ABC


class ProviderInterface(ABC):
    provider_name: str

    @classmethod
    def __init_subclass__(cls):
        """
        check that provider_name has been implemented on class initialization
        """
        if not hasattr(cls, "provider_name"):
            raise NotImplementedError("provider_name is required")
